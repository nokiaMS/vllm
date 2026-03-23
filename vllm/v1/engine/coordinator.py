# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy  # 导入深拷贝/浅拷贝工具模块
import multiprocessing  # 导入多进程模块
import time  # 导入时间模块
import weakref  # 导入弱引用模块，用于防止循环引用

import msgspec.msgpack  # 导入msgspec的msgpack序列化/反序列化模块
import zmq  # 导入ZeroMQ消息队列库

from vllm.config import ParallelConfig  # 从vllm配置中导入并行配置类
from vllm.logger import init_logger  # 导入日志初始化函数
from vllm.utils.network_utils import make_zmq_socket  # 导入ZMQ套接字创建工具函数
from vllm.utils.system_utils import get_mp_context, set_process_title  # 导入多进程上下文获取和进程标题设置工具
from vllm.v1.engine import EngineCoreOutputs, EngineCoreRequestType  # 导入引擎核心输出类和请求类型枚举
from vllm.v1.serial_utils import MsgpackDecoder  # 导入Msgpack解码器
from vllm.v1.utils import get_engine_client_zmq_addr, shutdown  # 导入ZMQ地址生成函数和关闭工具函数

logger = init_logger(__name__)  # 初始化当前模块的日志记录器


# [中文注释] DP Coordinator 管理类（在主进程中创建，启动 DPCoordinatorProc 子进程）。
#   提供三个 ZMQ 地址：
#     stats_publish_address — 前端订阅负载统计的 XPUB 地址
#     coord_in_address — 引擎向 Coordinator 发送统计和 wave 通知的 XPUB 地址
#     coord_out_address — Coordinator 从引擎接收消息的 PULL 地址
class DPCoordinator:
    """Coordinator process used for data-parallel deployments (DP>1).

    Intermediates between multiple DP engine rank processes and one or more
    front-end API server processes.

    * Collects stats from each DP engine (currently just waiting and running
      queue lengths), and publishes these to all front-ends for use in
      load-balancing decisions.

    * Keeps track of the current DP "request wave" number and running state
      of the engines. This is received from the DP rank 0 engine and published
      to the front-end processes along with the current load stats.

      The engines alternate between a global running/paused state. The global
      "request wave" number is a count of the number of times that the workers
      collectively move from a running state to a paused state. This transition
      is synchronized via the all-reduce operation performed in the
      DPEngineCoreProc._has_global_unfinished_reqs method.

    * Broadcasts the START_DP_WAVE message to engines to move them from paused
      to running state when one engine receives a new request. This can happen
      in two cases:
      1) A front-end sending a new request while the engines are paused will
         concurrently notify the coordinator.
      2) An engine receiving a request for a stale request wave while in paused
         state will notify the coordinator.

    Engines will move into running state when receiving a new request or
    START_DP_WAVE message.

    Note that when deployed in External LB mode, no stats will be published by
    the engines and thus updates will only be sent to front-ends when the
    request wave / running state changes.
    """

    def __init__(  # 构造函数定义
        self, parallel_config: ParallelConfig, enable_wave_coordination: bool = True  # 接收并行配置和是否启用wave协调
    ):  # 构造函数参数列表结束
        """初始化DP协调器，创建并启动协调器子进程。"""
        dp_size = parallel_config.data_parallel_size  # 获取数据并行大小
        assert dp_size > 1, "Coordinator only used for data parallel"  # 断言数据并行度大于1

        host = parallel_config.data_parallel_master_ip  # 获取数据并行主节点IP地址

        # Assume coordinator is colocated with front-end procs when not in
        # either external or hybrid DP LB mode.
        local_only = not parallel_config.local_engines_only  # 判断是否仅使用本地通信（非外部/混合负载均衡模式）
        front_publish_address = get_engine_client_zmq_addr(  # 生成前端发布地址
            local_only=local_only, host=host  # 传入本地模式标志和主机地址
        )  # 前端发布地址生成完成

        local_only_eng = dp_size == parallel_config.data_parallel_size_local  # 判断引擎是否全部在本地节点
        # NOTE(yongji): handling scaling from intra-node to inter-node
        if parallel_config.enable_elastic_ep:  # 如果启用了弹性专家并行
            local_only_eng = False  # 则不限制为本地通信（支持跨节点扩展）
        back_publish_address = get_engine_client_zmq_addr(local_only_eng, host)  # 生成后端发布地址（协调器到引擎）
        back_output_address = get_engine_client_zmq_addr(local_only_eng, host)  # 生成后端输出地址（引擎到协调器）

        context = get_mp_context()  # 获取多进程上下文（fork/spawn等）
        self.proc: multiprocessing.Process = context.Process(  # 创建协调器子进程
            target=DPCoordinatorProc.run_coordinator,  # 指定子进程入口函数
            name="VLLM_DP_Coordinator",  # 设置进程名称
            kwargs={  # 传递关键字参数
                "engine_count": parallel_config.data_parallel_size,  # 引擎数量
                "front_publish_address": front_publish_address,  # 前端发布地址
                "back_output_address": back_output_address,  # 后端输出地址
                "back_publish_address": back_publish_address,  # 后端发布地址
                "enable_wave_coordination": enable_wave_coordination,  # 是否启用wave协调
            },  # 关键字参数字典结束
            daemon=True,  # 设置为守护进程（主进程退出时自动终止）
        )  # 进程对象创建完成
        self.proc.start()  # 启动协调器子进程

        self.stats_publish_address = front_publish_address  # 保存统计信息发布地址（供前端订阅）
        self.coord_in_address = back_publish_address  # 保存协调器输入地址（引擎向协调器发送消息）
        self.coord_out_address = back_output_address  # 保存协调器输出地址（协调器从引擎接收消息）
        self._finalizer = weakref.finalize(self, shutdown, [self.proc])  # 注册弱引用析构器，确保进程被正确关闭

    def get_stats_publish_address(self) -> str:  # 获取统计信息发布地址的方法
        """获取前端订阅统计信息的ZMQ地址。"""
        return self.stats_publish_address  # 返回统计信息发布地址

    def get_engine_socket_addresses(self) -> tuple[str, str]:  # 获取引擎套接字地址的方法
        """返回ZMQ输入地址和输出地址的元组。"""
        return self.coord_in_address, self.coord_out_address  # 返回协调器输入和输出地址

    def shutdown(self, timeout: float | None = None) -> None:  # 关闭协调器进程的方法
        """关闭协调器进程，支持可配置的超时时间。"""
        if self._finalizer.detach() is not None:  # 如果析构器尚未被触发，则分离并执行关闭
            shutdown([self.proc], timeout=timeout)  # 调用shutdown工具函数关闭进程


# [中文注释] 单个引擎的状态：[waiting 队列中请求数, running 正在推理的请求数]
class EngineState:  # 引擎状态类定义
    """单个引擎的负载状态，记录等待和运行中的请求数量。"""
    def __init__(self):  # 构造函数
        self.request_counts = [0, 0]  # [waiting等待中请求数, running运行中请求数]


# [中文注释] DP Coordinator 子进程的核心实现。运行在独立进程中，通过 ZMQ 协调多引擎：
#   process_input_socket() 主循环同时监听三个 socket：
#     1. publish_front (XPUB) — 前端连接：接收 FIRST_REQ 唤醒通知和 SCALE_ELASTIC_EP 伸缩通知
#     2. publish_back (XPUB) — 引擎连接：管理引擎订阅/取消订阅
#     3. output_back (PULL) — 引擎输出：接收 scheduler_stats、wave_complete、start_wave 消息
#   定期（每 100ms）向所有前端发布 (counts, wave, running) 负载统计
#   Wave 协调：跟踪 current_wave 和 engines_running 状态，广播 START_DP_WAVE 唤醒所有引擎
class DPCoordinatorProc:
    """DP协调器子进程的核心实现，运行在独立进程中，通过ZMQ套接字协调多个引擎的状态和负载统计。"""

    def __init__(  # 构造函数定义
        self,  # 实例自身引用
        engine_count: int,  # 引擎数量
        min_stats_update_interval_ms: int = 100,  # 最小统计更新间隔（毫秒），默认100ms
        enable_wave_coordination: bool = True,  # 是否启用wave协调，默认启用
    ):  # 构造函数参数列表结束
        """初始化DP协调器进程，设置ZMQ上下文和引擎状态。"""
        set_process_title("DPCoordinator")  # 设置当前进程标题为"DPCoordinator"
        self.ctx = zmq.Context()  # 创建ZMQ上下文对象

        self.engines = [EngineState() for _ in range(engine_count)]  # 为每个引擎创建状态对象列表

        self.stats_update_interval_ms = min_stats_update_interval_ms  # 保存统计更新间隔配置
        self.enable_wave_coordination = enable_wave_coordination  # 保存是否启用wave协调的配置

    @staticmethod  # 静态方法装饰器（子进程入口函数，无需实例）
    def run_coordinator(  # 协调器运行入口函数
        engine_count: int,  # 引擎数量参数
        front_publish_address: str,  # 前端发布地址参数
        back_output_address: str,  # 后端输出地址参数
        back_publish_address: str,  # 后端发布地址参数
        min_stats_update_interval_ms: int = 100,  # 最小统计更新间隔（毫秒）
        enable_wave_coordination: bool = True,  # 是否启用wave协调
    ):  # 参数列表结束
        """协调器子进程的入口方法，创建实例并开始处理消息循环。"""
        coordinator = DPCoordinatorProc(  # 创建协调器进程实例
            engine_count=engine_count,  # 传入引擎数量
            min_stats_update_interval_ms=min_stats_update_interval_ms,  # 传入统计更新间隔
            enable_wave_coordination=enable_wave_coordination,  # 传入wave协调开关
        )  # 协调器实例创建完成
        try:  # 尝试运行主循环
            coordinator.process_input_socket(  # 调用主消息处理循环
                front_publish_address,  # 传入前端发布地址
                back_output_address,  # 传入后端输出地址
                back_publish_address,  # 传入后端发布地址
            )  # 主循环调用结束
        except KeyboardInterrupt:  # 捕获键盘中断异常（Ctrl+C）
            logger.info("DP Coordinator process exiting")  # 记录协调器进程退出日志

    def process_input_socket(  # 主消息处理循环方法定义
        self,  # 实例自身引用
        front_publish_address: str,  # 前端发布地址
        back_output_address: str,  # 后端输出地址
        back_publish_address: str,  # 后端发布地址
    ):  # 参数列表结束
        """主消息处理循环，监听前端和引擎的ZMQ套接字，协调DP状态。"""
        decoder = MsgpackDecoder(EngineCoreOutputs)  # 创建EngineCoreOutputs类型的Msgpack解码器

        # For tracking request wave progression.
        current_wave = 0  # 当前请求wave编号，初始为0
        engines_running = False  # 引擎是否处于运行状态，初始为暂停

        # For tracking request counts for internal load-balancing.
        stats_changed = False  # 统计信息是否有变化的标志
        last_stats_step = -1  # 上一次统计的步骤计数器
        last_stats_wave = -1  # 上一次统计的wave编号
        last_step_counts: list[list[int]] | None = None  # 上一步的引擎计数快照，用于发布

        with (  # 使用上下文管理器同时创建三个ZMQ套接字
            make_zmq_socket(  # 创建前端发布套接字
                path=front_publish_address,  # IPC地址，用于向前端发布统计信息
                ctx=self.ctx,  # 使用协调器的ZMQ上下文
                socket_type=zmq.XPUB,  # XPUB类型，支持订阅过滤和双向通信
                bind=True,  # 绑定模式（服务端）
            ) as publish_front,  # 命名为publish_front（前端发布套接字）
            make_zmq_socket(  # 创建后端输出套接字
                path=back_output_address,  # IPC或TCP地址，用于接收引擎输出
                ctx=self.ctx,  # 使用协调器的ZMQ上下文
                socket_type=zmq.PULL,  # PULL类型，从多个引擎拉取消息
                bind=True,  # 绑定模式（服务端）
            ) as output_back,  # 命名为output_back（后端输出套接字）
            make_zmq_socket(  # 创建后端发布套接字
                path=back_publish_address,  # IPC或TCP地址，用于向引擎广播消息
                ctx=self.ctx,  # 使用协调器的ZMQ上下文
                socket_type=zmq.XPUB,  # XPUB类型，支持订阅管理
                bind=True,  # 绑定模式（服务端）
            ) as publish_back,  # 命名为publish_back（后端发布套接字）
        ):  # with语句块开始
            # Wait until all engines subscribe.
            for _ in self.engines:  # 遍历所有引擎，等待每个引擎的订阅消息
                if publish_back.recv() != b"\x01":  # 如果收到的不是订阅消息（\x01表示订阅）
                    logger.error(  # 记录错误日志
                        "DP Coordinator received unexpected message while "  # 错误消息第一行
                        "waiting for engines to subscribe"  # 错误消息第二行
                    )  # 日志记录结束
                    return  # 收到异常消息，直接返回退出
            # Send ready message to engines.
            publish_back.send(b"READY")  # 向所有引擎广播READY就绪消息

            logger.info("All engine subscriptions received by DP coordinator")  # 记录所有引擎订阅完成的日志

            poller = zmq.Poller()  # 创建ZMQ轮询器，用于同时监听多个套接字
            poller.register(publish_front, zmq.POLLIN)  # 注册前端发布套接字到轮询器（监听输入事件）
            poller.register(publish_back, zmq.POLLIN)  # 注册后端发布套接字到轮询器（监听输入事件）
            poller.register(output_back, zmq.POLLIN)  # 注册后端输出套接字到轮询器（监听输入事件）
            last_publish_time = 0  # 上一次发布统计信息的时间戳（毫秒），初始为0
            while True:  # 主消息处理无限循环
                elapsed = int(time.time() * 1000) - last_publish_time  # 计算距离上次发布已过去的毫秒数
                # Send at stats_update_interval_ms interval if the stats have
                # changed, or otherwise every 5 seconds.
                wait_for = self.stats_update_interval_ms if stats_changed else 5000  # 统计有变化时按配置间隔发布，否则每5秒发布一次

                # Wait at least 50ms to ensure we've received all stats for
                # the current step.
                min_timeout = 50 if last_step_counts is None else 0  # 如果没有缓存的步骤计数，至少等待50ms收集完整统计

                events = poller.poll(timeout=max(min_timeout, wait_for - elapsed))  # 轮询所有注册的套接字，等待事件或超时
                if not events:  # 如果轮询超时（没有收到任何事件）
                    # Poller timeout - publish current stats to front-ends.
                    if last_step_counts is not None:  # 如果有缓存的上一步计数快照
                        engine_req_counts_list = last_step_counts  # 使用缓存的计数数据
                        last_step_counts = None  # 清除缓存
                    else:  # 否则没有缓存的计数
                        engine_req_counts_list = self._get_engine_counts()  # 获取当前各引擎的请求计数
                        stats_changed = False  # 重置统计变化标志

                    to_publish = (engine_req_counts_list, current_wave, engines_running)  # 构建要发布的数据元组
                    publish_front.send(msgspec.msgpack.encode(to_publish))  # 将数据序列化后发送给前端
                    last_publish_time = int(time.time() * 1000)  # 更新上次发布时间
                    continue  # 跳过本次循环的其余部分，继续下一轮

                events = dict(events)  # 将轮询结果转换为字典，键为套接字对象
                wave_state_changed = False  # 重置wave状态变化标志

                if publish_back in events:  # 如果后端发布套接字有事件（引擎订阅/取消订阅）
                    buffer = publish_back.recv()  # 接收来自引擎的消息
                    if buffer == b"\x01":  # 如果是订阅消息（新引擎加入）
                        # NOTE(yongji): newly started engine subscribed
                        # We need to send READY message here instead of receiving
                        # SCALE_ELASTIC_EP notification from engine core client
                        # as SCALE_ELASTIC_EP is only sent when
                        # new engines finished initialization.
                        # Subscription message, on the other hand, is sent
                        # by each engine during initialization
                        publish_back.send(b"READY")  # 向新订阅的引擎发送READY就绪消息
                    elif buffer != b"\x00":  # 如果不是取消订阅消息（\x00），则是意外消息
                        logger.error(  # 记录错误日志
                            "DP Coordinator received unexpected message from engines"  # 错误信息：收到引擎的意外消息
                        )  # 日志记录结束

                if publish_front in events:  # 如果前端发布套接字有事件（前端消息）
                    buffer = publish_front.recv()  # 接收来自前端的消息
                    if buffer in (b"\x01", b"\x00"):  # 如果是订阅或取消订阅消息
                        # Ignore subscription messages.
                        continue  # 忽略订阅消息，继续下一轮循环

                    decoded = msgspec.msgpack.decode(buffer)  # 反序列化接收到的消息
                    if (  # 判断是否为弹性EP扩缩容通知
                        isinstance(decoded, (list, tuple))  # 检查解码结果是否为列表或元组
                        and len(decoded) == 2  # 检查长度是否为2
                        and decoded[0] == "SCALE_ELASTIC_EP"  # 检查第一个元素是否为扩缩容标识
                    ):  # 条件判断结束
                        # Handle scale up notification
                        new_engine_count = decoded[1]  # 获取新的引擎数量
                        current_count = len(self.engines)  # 获取当前引擎数量
                        if new_engine_count > current_count:  # 如果需要扩容（新数量大于当前数量）
                            for _ in range(new_engine_count - current_count):  # 遍历需要新增的引擎数
                                self.engines.append(EngineState())  # 为每个新引擎添加状态对象
                            # NOTE(yongji): handle the case
                            # where newly started engines have current_wave = 0
                            # if existing engines just finished a wave
                            # and engine_running isn't updated yet at
                            # CoordinatorProc requests routed to newly started
                            # engines may not wake up existing engines, as long
                            # as 0 < request.wave < existing engines'
                            # current_wave
                            # we note that 0 is the wave number for the new
                            # engine
                            logger.info(  # 记录扩容日志
                                "DPCoordinator scaled up from %s to %s engines",  # 日志格式字符串
                                current_count,  # 原引擎数量
                                new_engine_count,  # 新引擎数量
                            )  # 日志记录结束
                        else:  # 否则需要缩容（新数量小于等于当前数量）
                            self.engines = self.engines[:new_engine_count]  # 截断引擎列表到新数量
                            logger.info(  # 记录缩容日志
                                "DPCoordinator scaled down from %s to %s engines",  # 日志格式字符串
                                current_count,  # 原引擎数量
                                new_engine_count,  # 新引擎数量
                            )  # 日志记录结束
                        continue  # 跳过后续的引擎通知处理，继续下一轮循环

                    # Wave coordination: handle new-request messages from front-end.
                    # Only process these when wave coordination is enabled
                    if self.enable_wave_coordination:  # 如果启用了wave协调
                        # We received a message on the front-end XPUB socket,
                        # from an API server sending a new request while the
                        # engines are paused, so that we can wake the other
                        # engines.
                        engine_to_exclude, wave = decoded  # 解构前端消息：要排除的引擎索引和wave编号
                        if not engines_running:  # 如果引擎当前处于暂停状态
                            if wave < current_wave:  # 如果前端发送的wave编号已过期
                                # If the wave number is stale, ensure the message
                                # is handled by all the engines.
                                engine_to_exclude = None  # 清除排除引擎，让所有引擎都处理此消息

                            engines_running = True  # 将引擎状态设置为运行中
                            wave_state_changed = True  # 标记wave状态已变化
                            self._send_start_wave(  # 向所有引擎广播启动wave消息
                                publish_back, current_wave, engine_to_exclude  # 传入后端套接字、当前wave和排除引擎
                            )  # 广播调用结束

                if output_back in events:  # 如果后端输出套接字有事件（引擎发送的消息）
                    # We received a message from one of the engines.

                    buffer = output_back.recv()  # 接收来自引擎的原始字节消息
                    outputs: EngineCoreOutputs = decoder.decode(buffer)  # 反序列化为EngineCoreOutputs对象

                    assert not outputs.outputs  # 断言没有常规输出（协调器只处理统计和wave消息）
                    assert outputs.utility_output is None  # 断言没有工具输出

                    eng_index = outputs.engine_index  # 获取发送消息的引擎索引
                    scheduler_stats = outputs.scheduler_stats  # 获取调度器统计信息
                    if scheduler_stats:  # 如果包含调度器统计信息
                        # 1. Updated request load stats - update our local
                        # state with these.
                        stats = self.engines[eng_index].request_counts  # 获取该引擎的请求计数引用
                        stats_step = scheduler_stats.step_counter  # 获取统计中的步骤计数器
                        stats_wave = scheduler_stats.current_wave  # 获取统计中的wave编号
                        if (  # 判断是否为更新的统计数据（wave更大，或同wave但步骤更大）
                            stats_wave > last_stats_wave  # wave编号比上次记录的大
                            or stats_wave == last_stats_wave  # 或wave相同
                            and stats_step > last_stats_step  # 且步骤计数器比上次大
                        ):  # 条件判断结束
                            if stats_changed:  # 如果之前已有统计变化（需要保存旧数据快照）
                                last_step_counts = self._get_engine_counts(do_copy=True)  # 复制当前计数作为上一步快照
                            last_stats_step = stats_step  # 更新上次统计步骤
                            last_stats_wave = stats_wave  # 更新上次统计wave
                        elif stats_wave != last_stats_wave or (  # 否则如果wave不同或步骤不同（乱序统计）
                            stats_step != last_stats_step  # 步骤计数器不一致
                        ):  # 乱序条件判断结束
                            logger.warning(  # 记录乱序统计的警告日志
                                "Received stats for out-of-order "  # 警告消息第一行
                                "step (%d, %d) from engine %d (expected "  # 警告消息第二行
                                "> (%d, %d))",  # 警告消息第三行
                                stats_wave,  # 实际的wave编号
                                stats_step,  # 实际的步骤计数器
                                eng_index,  # 发送统计的引擎索引
                                last_stats_wave,  # 期望的wave编号
                                last_stats_step,  # 期望的步骤计数器
                            )  # 警告日志结束
                        stats[0] = scheduler_stats.num_waiting_reqs  # 更新该引擎的等待请求数
                        stats[1] = scheduler_stats.num_running_reqs  # 更新该引擎的运行请求数
                        stats_changed = True  # 标记统计信息已变化

                    # Wave coordination: handle wave completion and start notifications
                    # Only process these when wave coordination is enabled
                    if self.enable_wave_coordination:  # 如果启用了wave协调机制
                        if (wave := outputs.wave_complete) is not None:  # 如果收到wave完成通知（海象运算符赋值）
                            # 2. Notification from rank 0 engine that we've
                            # moved into the global paused state
                            # (engines_running==False).
                            if current_wave <= wave:  # 如果当前wave不超过已完成的wave
                                new_wave = wave + 1  # 计算新的wave编号（已完成wave加1）
                                logger.debug(  # 记录wave推进的调试日志
                                    "Moving DP wave from %d to %d.",  # 日志格式字符串
                                    current_wave,  # 旧的wave编号
                                    new_wave,  # 新的wave编号
                                )  # 调试日志结束
                                current_wave = new_wave  # 更新当前wave编号
                                engines_running = False  # 将引擎状态设置为暂停
                                wave_state_changed = True  # 标记wave状态已变化
                        elif (wave := outputs.start_wave) is not None and (  # 如果收到启动wave通知且满足条件
                            wave > current_wave  # wave编号大于当前wave
                            or (wave == current_wave and not engines_running)  # 或wave相同但引擎处于暂停状态
                        ):  # 条件判断结束
                            # 3. The engine received request for a non-current wave
                            # so we must ensure that other engines progress to the
                            # next wave (race condition handling).
                            logger.debug(  # 记录启动wave的调试日志
                                "Starting wave %d after notification of "  # 日志格式字符串第一行
                                "stale wave request from engine.",  # 日志格式字符串第二行
                                wave,  # 要启动的wave编号
                            )  # 调试日志结束
                            current_wave = wave  # 更新当前wave编号
                            engines_running = True  # 将引擎状态设置为运行中
                            wave_state_changed = True  # 标记wave状态已变化
                            self._send_start_wave(publish_back, wave, eng_index)  # 广播启动wave消息给其他引擎

                if wave_state_changed:  # 如果wave状态在本轮循环中发生了变化
                    message = (None, current_wave, engines_running)  # 构建wave状态更新消息（无统计数据、当前wave、运行状态）
                    publish_front.send(msgspec.msgpack.encode(message))  # 将wave状态变化序列化后发送给前端

    @staticmethod  # 静态方法装饰器
    def _send_start_wave(  # 发送启动wave广播消息的方法
        socket: zmq.Socket, wave: int, exclude_engine_index: int | None  # 套接字、wave编号、要排除的引擎索引
    ):  # 参数列表结束
        """向所有引擎广播START_DP_WAVE消息，包含当前wave编号和已收到请求的引擎索引（该引擎无需额外通知）。"""
        wave_encoded = msgspec.msgpack.encode((wave, exclude_engine_index))  # 将wave编号和排除引擎索引序列化为msgpack格式
        socket.send_multipart((EngineCoreRequestType.START_DP_WAVE.value, wave_encoded))  # 通过多部分消息发送启动wave命令

    def _get_engine_counts(self, do_copy=False) -> list[list[int]]:  # 获取所有引擎请求计数的方法
        """返回每个引擎的[等待中, 运行中]请求计数列表。"""
        if do_copy:  # 如果需要深拷贝（避免数据竞争）
            return [copy.copy(e.request_counts) for e in self.engines]  # 返回每个引擎请求计数的浅拷贝列表
        return [e.request_counts for e in self.engines]  # 返回每个引擎请求计数的直接引用列表
