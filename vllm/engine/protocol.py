# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 以上为 Apache-2.0 开源许可证声明和版权声明

from abc import ABC, abstractmethod  # 导入抽象基类（ABC）和抽象方法装饰器，用于定义接口
from collections.abc import AsyncGenerator, Iterable, Mapping  # 导入异步生成器、可迭代对象和映射的抽象类型
from dataclasses import dataclass  # 导入数据类装饰器
from typing import TYPE_CHECKING, Any  # 导入类型检查标志和 Any 类型

from vllm.config import ModelConfig, VllmConfig  # 导入模型配置和 vLLM 总配置类
from vllm.distributed.weight_transfer.base import (  # 导入权重传输相关的请求类
    WeightTransferInitRequest,  # 权重传输初始化请求
    WeightTransferUpdateRequest,  # 权重传输更新请求
)
from vllm.inputs.data import ProcessorInputs, PromptType  # 导入处理器输入和提示类型
from vllm.lora.request import LoRARequest  # 导入 LoRA 请求类
from vllm.outputs import PoolingRequestOutput, RequestOutput  # 导入池化请求输出和通用请求输出类
from vllm.plugins.io_processors import IOProcessor  # 导入 IO 处理器接口
from vllm.pooling_params import PoolingParams  # 导入池化参数类
from vllm.renderers import BaseRenderer  # 导入基础渲染器类
from vllm.sampling_params import SamplingParams  # 导入采样参数类
from vllm.tasks import SupportedTask  # 导入支持的任务类型
from vllm.v1.engine import EngineCoreRequest  # 导入引擎核心请求类
from vllm.v1.engine.input_processor import InputProcessor  # 导入输入处理器类

# 仅在类型检查时导入以下模块（避免循环导入和运行时开销）
if TYPE_CHECKING:
    from vllm.v1.engine import PauseMode  # 导入暂停模式类型（仅用于类型注解）


@dataclass  # 数据类装饰器，自动生成 __init__、__repr__ 等方法
class StreamingInput:
    """流式生成请求的输入数据类。

    用于 generate() 方法以支持多轮流式会话，
    其中输入通过异步生成器逐步提供。

    属性:
        prompt: 处理器输入数据（包含 token IDs 和多模态数据等）
        sampling_params: 采样参数（温度、top_p 等），可选
    """

    prompt: ProcessorInputs  # 处理器输入数据
    sampling_params: SamplingParams | None = None  # 采样参数，None 表示使用默认值


class EngineClient(ABC):
    """引擎客户端协议类（抽象基类）。

    定义了客户端与 vLLM 引擎之间的通信接口。
    所有引擎客户端实现（如 AsyncLLM）都必须实现此接口中定义的抽象方法。
    该类规定了引擎的核心功能，包括生成、编码、中止请求、
    健康检查、性能分析、缓存管理、睡眠/唤醒、LoRA 管理等。
    """

    vllm_config: VllmConfig  # vLLM 总配置对象
    model_config: ModelConfig  # 模型配置对象
    renderer: BaseRenderer  # 渲染器，负责将聊天模板转换为模型输入
    io_processor: IOProcessor | None  # IO 处理器插件，可选
    input_processor: InputProcessor  # 输入处理器，负责预处理输入数据

    @property  # 属性装饰器，将方法转换为只读属性
    @abstractmethod  # 抽象方法装饰器，子类必须实现
    def is_running(self) -> bool:
        """检查引擎是否正在运行。

        返回:
            如果引擎正在运行则返回 True
        """
        ...

    @property  # 属性装饰器
    @abstractmethod  # 抽象方法装饰器
    def is_stopped(self) -> bool:
        """检查引擎是否已停止。

        返回:
            如果引擎已停止则返回 True
        """
        ...

    @property  # 属性装饰器
    @abstractmethod  # 抽象方法装饰器
    def errored(self) -> bool:
        """检查引擎是否处于错误状态。

        返回:
            如果引擎出错则返回 True
        """
        ...

    @property  # 属性装饰器
    @abstractmethod  # 抽象方法装饰器
    def dead_error(self) -> BaseException:
        """获取导致引擎终止的异常。

        返回:
            导致引擎死亡的异常对象
        """
        ...

    @abstractmethod  # 抽象方法装饰器，子类必须实现
    def generate(
        self,
        prompt: EngineCoreRequest  # 引擎核心请求对象
        | PromptType  # 或提示类型（字符串、token ID 列表等）
        | ProcessorInputs  # 或处理器输入
        | AsyncGenerator[StreamingInput, None],  # 或流式输入的异步生成器
        sampling_params: SamplingParams,  # 采样参数（温度、top_p、max_tokens 等）
        request_id: str,  # 请求的唯一标识符
        *,  # 以下为仅限关键字参数
        prompt_text: str | None = None,  # 原始提示文本（用于日志记录）
        lora_request: LoRARequest | None = None,  # LoRA 适配器请求
        tokenization_kwargs: dict[str, Any] | None = None,  # 分词器额外参数
        trace_headers: Mapping[str, str] | None = None,  # 追踪头信息
        priority: int = 0,  # 请求优先级（越大越优先）
        data_parallel_rank: int | None = None,  # 指定处理该请求的数据并行排名
        reasoning_ended: bool | None = None,  # 推理阶段是否已结束
    ) -> AsyncGenerator[RequestOutput, None]:
        """为请求生成输出（文本生成）。

        这是引擎的核心生成方法，接受各种形式的输入提示，
        使用指定的采样参数生成文本，并以异步生成器的形式
        逐步返回生成结果。

        参数:
            prompt: 输入提示，支持多种格式
            sampling_params: 采样参数
            request_id: 请求唯一 ID
            prompt_text: 原始提示文本
            lora_request: LoRA 请求
            tokenization_kwargs: 分词参数
            trace_headers: 追踪头
            priority: 优先级
            data_parallel_rank: 数据并行排名
            reasoning_ended: 推理是否结束
        返回:
            异步生成器，逐步产出 RequestOutput 对象
        """
        ...

    @abstractmethod  # 抽象方法装饰器
    def encode(
        self,
        prompt: PromptType | ProcessorInputs,  # 输入提示（字符串、token ID 列表或处理器输入）
        pooling_params: PoolingParams,  # 池化参数
        request_id: str,  # 请求唯一标识符
        lora_request: LoRARequest | None = None,  # LoRA 适配器请求
        trace_headers: Mapping[str, str] | None = None,  # 追踪头信息
        priority: int = 0,  # 请求优先级
        tokenization_kwargs: dict[str, Any] | None = None,  # 分词器额外参数
        reasoning_ended: bool | None = None,  # 推理阶段是否已结束
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """为池化模型的请求生成输出（嵌入/分类/评分等）。

        用于嵌入模型、重排序模型等池化类型的模型，
        将输入文本转换为向量表示或分类结果。

        参数:
            prompt: 输入提示
            pooling_params: 池化参数
            request_id: 请求唯一 ID
            lora_request: LoRA 请求
            trace_headers: 追踪头
            priority: 优先级
            tokenization_kwargs: 分词参数
            reasoning_ended: 推理是否结束
        返回:
            异步生成器，产出 PoolingRequestOutput 对象
        """
        ...

    @abstractmethod  # 抽象方法装饰器
    async def abort(self, request_id: str | Iterable[str]) -> None:
        """中止一个或多个请求。

        参数:
            request_id: 要中止的请求的唯一 ID，
                        或包含多个 ID 的可迭代对象。
        """
        ...

    @abstractmethod  # 抽象方法装饰器
    async def is_tracing_enabled(self) -> bool:
        """检查追踪功能是否已启用。

        返回:
            如果追踪已启用则返回 True
        """
        ...

    @abstractmethod  # 抽象方法装饰器
    async def do_log_stats(self) -> None:
        """执行统计日志记录。

        收集并记录引擎运行的统计信息。
        """
        ...

    @abstractmethod  # 抽象方法装饰器
    async def check_health(self) -> None:
        """健康检查。

        检查引擎是否正常运行，如果不健康则抛出异常。
        """
        ...

    @abstractmethod  # 抽象方法装饰器
    async def start_profile(self) -> None:
        """开始性能分析。

        启动引擎的性能分析器，开始收集性能数据。
        """
        ...

    @abstractmethod  # 抽象方法装饰器
    async def stop_profile(self) -> None:
        """停止性能分析。

        停止引擎的性能分析器，结束性能数据收集。
        """
        ...

    @abstractmethod  # 抽象方法装饰器
    async def reset_mm_cache(self) -> None:
        """重置多模态缓存。

        清除多模态处理器的缓存数据。
        """
        ...

    @abstractmethod  # 抽象方法装饰器
    async def reset_encoder_cache(self) -> None:
        """重置编码器缓存。

        清除编码器（如视觉编码器）的缓存数据。
        """
        ...

    @abstractmethod  # 抽象方法装饰器
    async def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """重置前缀缓存，并可选择重置连接器缓存。

        参数:
            reset_running_requests: 是否同时重置正在运行的请求的缓存
            reset_connector: 是否同时重置已配置的连接器缓存
        返回:
            重置是否成功
        """
        ...

    @abstractmethod  # 抽象方法装饰器
    async def sleep(self, level: int = 1, mode: "PauseMode" = "abort") -> None:
        """使引擎进入睡眠模式。

        睡眠模式下引擎会释放 GPU 内存，
        适用于需要临时腾出 GPU 资源的场景。

        参数:
            level: 睡眠级别（1 为标准睡眠）
            mode: 暂停模式（"abort" 中止请求 / "wait" 等待完成 / "keep" 保持）
        """
        ...

    @abstractmethod  # 抽象方法装饰器
    async def wake_up(self, tags: list[str] | None = None) -> None:
        """唤醒引擎。

        从睡眠模式中恢复引擎，重新加载模型权重到 GPU。

        参数:
            tags: 可选的标签列表，用于标识唤醒来源
        """
        ...

    @abstractmethod  # 抽象方法装饰器
    async def is_sleeping(self) -> bool:
        """检查引擎是否处于睡眠状态。

        返回:
            如果引擎正在睡眠则返回 True
        """
        ...

    @abstractmethod  # 抽象方法装饰器
    async def add_lora(self, lora_request: LoRARequest) -> bool:
        """加载新的 LoRA 适配器到引擎中。

        将指定的 LoRA 适配器加载到引擎中，以便后续请求使用。

        参数:
            lora_request: LoRA 适配器请求对象
        返回:
            加载是否成功
        """
        ...

    @abstractmethod  # 抽象方法装饰器
    async def pause_generation(
        self,
        *,  # 以下为仅限关键字参数
        mode: "PauseMode" = "abort",  # 暂停模式
        wait_for_inflight_requests: bool = False,  # 是否等待进行中的请求（已弃用）
        clear_cache: bool = True,  # 是否清除缓存（已弃用）
    ) -> None:
        """暂停新的生成/编码请求。

        暂停引擎接受新请求，并根据指定的模式处理正在进行的请求。

        参数:
            mode: 如何处理正在进行的请求:
                - ``"abort"``: 立即中止所有进行中的请求，
                  返回带有 "abort" 原因的部分结果（默认）。
                - ``"wait"``: 等待进行中的请求完成。
                - ``"keep"``: 冻结队列中的请求；在调用
                  :meth:`resume_generation` 时恢复。
            wait_for_inflight_requests: 已弃用。请改用 ``mode="wait"``。
            clear_cache: 已弃用。排空后是否清除 KV 和前缀缓存。
        """
        ...

    @abstractmethod  # 抽象方法装饰器
    async def resume_generation(self) -> None:
        """恢复接受生成/编码请求。

        在调用 pause_generation() 暂停后，调用此方法恢复正常服务。
        """
        ...

    @abstractmethod  # 抽象方法装饰器
    async def is_paused(self) -> bool:
        """检查引擎是否当前处于暂停状态。

        返回:
            如果引擎已暂停则返回 True
        """
        ...

    @abstractmethod  # 抽象方法装饰器
    def shutdown(self, timeout: float | None = None) -> None:
        """关闭引擎。

        优雅地关闭引擎，可选择设置超时时间。

        参数:
            timeout: 关闭超时时间（秒），None 表示无限等待
        """
        ...

    async def scale_elastic_ep(
        self, new_data_parallel_size: int, drain_timeout: int = 300
    ) -> None:
        """弹性扩缩容引擎的专家并行规模。

        动态调整数据并行副本的数量，支持在线扩缩容。

        参数:
            new_data_parallel_size: 新的数据并行大小
            drain_timeout: 排空超时时间（秒），默认 300 秒
        """
        raise NotImplementedError  # 默认未实现，需要子类覆盖

    async def collective_rpc(
        self,
        method: str,  # RPC 方法路径
        timeout: float | None = None,  # 超时时间（秒）
        args: tuple = (),  # 位置参数
        kwargs: dict | None = None,  # 关键字参数
    ):
        """对所有工作器执行集体 RPC 调用。

        向所有分布式工作器广播执行指定的方法调用。

        参数:
            method: 要调用的方法名称
            timeout: 调用超时时间（秒）
            args: 传递给方法的位置参数
            kwargs: 传递给方法的关键字参数
        """
        raise NotImplementedError  # 默认未实现，需要子类覆盖

    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        """获取引擎支持的任务类型。

        返回:
            支持的任务类型元组（如 generate、embed、classify 等）
        """
        raise NotImplementedError  # 默认未实现，需要子类覆盖

    async def init_weight_transfer_engine(
        self, init_request: WeightTransferInitRequest  # 权重传输初始化请求
    ) -> None:
        """初始化权重传输引擎（用于强化学习训练）。

        建立权重传输通道，允许在训练过程中更新模型权重。

        参数:
            init_request: 权重传输初始化请求，包含连接配置等信息
        """
        raise NotImplementedError  # 默认未实现，需要子类覆盖

    async def update_weights(self, request: WeightTransferUpdateRequest) -> None:
        """批量更新模型权重（用于强化学习训练）。

        接收来自训练进程的权重更新，并应用到推理模型中。

        参数:
            request: 权重传输更新请求，包含要更新的权重数据
        """
        raise NotImplementedError  # 默认未实现，需要子类覆盖
