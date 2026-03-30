# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
import importlib.util  # 导入模块查找工具，用于在不直接导入的情况下定位模块
import os  # 导入操作系统接口模块，用于环境变量和路径操作


def _get_torch_cuda_version():
    """获取 PyTorch 绑定的 CUDA 版本号。

    这是 _maybe_set_cuda_compatibility_path() 的辅助函数。
    不能通过直接导入 PyTorch 来获取版本号，因为直接导入会触发 CUDA 初始化，
    导致无法在初始化之前设置 LD_LIBRARY_PATH 环境变量。
    """
    try:  # 尝试查找 torch 模块信息
        spec = importlib.util.find_spec("torch")  # 查找 torch 模块的规格说明，不触发导入
        if not spec:  # 如果未找到 torch 模块
            return None  # 返回 None 表示未找到
        if spec.origin:  # 如果模块有 origin 属性（即模块文件路径）
            torch_root = os.path.dirname(spec.origin)  # 获取 torch 安装根目录
        elif spec.submodule_search_locations:  # 如果模块有子模块搜索路径
            torch_root = spec.submodule_search_locations[0]  # 取第一个子模块搜索路径作为根目录
        else:  # 两者都没有
            return None  # 无法确定 torch 根目录，返回 None
        version_path = os.path.join(torch_root, "version.py")  # 拼接 version.py 的完整路径
        if not os.path.exists(version_path):  # 如果 version.py 文件不存在
            return None  # 返回 None 表示无法获取版本
        # Load the version module without importing torch
        ver_spec = importlib.util.spec_from_file_location("torch.version", version_path)  # 从文件路径创建模块规格，避免导入整个 torch
        if not ver_spec or not ver_spec.loader:  # 如果规格或加载器无效
            return None  # 返回 None 表示无法加载版本模块
        module = importlib.util.module_from_spec(ver_spec)  # 根据规格创建模块对象
        # Avoid registering in sys.modules to not confuse future imports
        ver_spec.loader.exec_module(module)  # 执行模块代码以填充模块属性，但不注册到 sys.modules
        return getattr(module, "cuda", None)  # 返回模块中的 cuda 属性（即 CUDA 版本字符串），不存在则返回 None
    except Exception:  # 捕获所有异常
        return None  # 出错时返回 None，确保不中断程序


def _maybe_set_cuda_compatibility_path():
    """在启用 CUDA 前向兼容时设置 LD_LIBRARY_PATH 环境变量。

    必须在 'import torch' 之前运行，因为 torch 在导入时会加载 CUDA 共享库，
    而动态链接器只在库首次加载时查询 LD_LIBRARY_PATH。

    CUDA 前向兼容仅在部分专业级和数据中心级 NVIDIA GPU 上受支持。
    消费级 GPU（GeForce、RTX）不支持此功能，加载兼容库会导致 Error 803。
    """
    enable = os.environ.get("VLLM_ENABLE_CUDA_COMPATIBILITY", "0").strip().lower() in (  # 从环境变量读取是否启用 CUDA 兼容模式
        "1",  # 字符串 "1" 表示启用
        "true",  # 字符串 "true" 表示启用
    )
    if not enable:  # 如果未启用 CUDA 兼容模式
        return  # 直接返回，不做任何操作

    cuda_compat_path = os.environ.get("VLLM_CUDA_COMPATIBILITY_PATH", "")  # 从环境变量获取用户指定的 CUDA 兼容库路径
    if not cuda_compat_path or not os.path.isdir(cuda_compat_path):  # 如果用户未指定路径或路径不存在
        conda_prefix = os.environ.get("CONDA_PREFIX", "")  # 尝试获取 Conda 环境前缀路径
        conda_compat = os.path.join(conda_prefix, "cuda-compat")  # 拼接 Conda 环境下的 cuda-compat 目录路径
        if conda_prefix and os.path.isdir(conda_compat):  # 如果 Conda 前缀存在且 cuda-compat 目录存在
            cuda_compat_path = conda_compat  # 使用 Conda 环境下的兼容库路径
    if not cuda_compat_path or not os.path.isdir(cuda_compat_path):  # 如果仍然没有有效的兼容库路径
        torch_cuda_version = _get_torch_cuda_version()  # 获取 PyTorch 绑定的 CUDA 版本号
        if torch_cuda_version:  # 如果成功获取到 CUDA 版本号
            default_path = f"/usr/local/cuda-{torch_cuda_version}/compat"  # 构造默认的 CUDA 兼容库路径
            if os.path.isdir(default_path):  # 如果默认路径目录存在
                cuda_compat_path = default_path  # 使用默认路径
    if not cuda_compat_path or not os.path.isdir(cuda_compat_path):  # 如果最终仍无有效路径
        return  # 放弃设置，直接返回

    norm_path = os.path.normpath(cuda_compat_path)  # 规范化路径，统一路径格式
    existing = os.environ.get("LD_LIBRARY_PATH", "")  # 获取当前 LD_LIBRARY_PATH 环境变量的值
    ld_paths = existing.split(os.pathsep) if existing else []  # 按路径分隔符拆分为路径列表

    if ld_paths and ld_paths[0] and os.path.normpath(ld_paths[0]) == norm_path:  # 如果兼容库路径已经在 LD_LIBRARY_PATH 最前面
        return  # Already at the front  # 已经在最前面，无需重复设置

    new_paths = [norm_path] + [  # 将兼容库路径放到最前面，并去除原有列表中的重复项
        p for p in ld_paths if not p or os.path.normpath(p) != norm_path  # 过滤掉已存在的相同路径
    ]
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(new_paths)  # 更新 LD_LIBRARY_PATH 环境变量


_maybe_set_cuda_compatibility_path()  # 在导入 torch 之前调用，确保 CUDA 兼容库路径已设置

import torch  # 导入 PyTorch，此时 LD_LIBRARY_PATH 已正确设置

from vllm.logger import init_logger  # 从 vllm 导入日志初始化函数
from vllm.utils.torch_utils import is_torch_equal  # 从 vllm 导入 PyTorch 版本比较工具函数

logger = init_logger(__name__)  # 初始化当前模块的日志记录器

# set some common config/environment variables that should be set
# for all processes created by vllm and all processes
# that interact with vllm workers.
# they are executed whenever `import vllm` is called.

# see https://github.com/vllm-project/vllm/pull/15951
# it avoids unintentional cuda initialization from torch.cuda.is_available()
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"  # 设置 PyTorch 使用 NVML 检测 CUDA，避免意外初始化 CUDA

# see https://github.com/vllm-project/vllm/issues/10480
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"  # 将 TorchInductor 编译线程数限制为 1，避免多线程编译问题
# see https://github.com/vllm-project/vllm/issues/10619
torch._inductor.config.compile_threads = 1  # 在运行时配置中同样将编译线程数设为 1

# ===================================================
# torch 2.9 Inductor PythonWrapperCodegen monkeypatch
# ===================================================
# This change monkeypatches memory_plan_reuse in pytorch 2.9.0 to work around
# a test failure for test_multi_graph_piecewise_compile_outputs_equal.
# For more context, see https://github.com/pytorch/pytorch/pull/165514.


def memory_plan_reuse_patched(self):
    """修补后的内存计划复用函数。

    对 PyTorch 2.9.0 中的 memory_plan_reuse 进行猴子补丁，
    修复 test_multi_graph_piecewise_compile_outputs_equal 测试失败的问题。
    详情参见 https://github.com/pytorch/pytorch/pull/165514。
    """
    import torch._inductor.ir as ir  # 导入 Inductor 中间表示模块
    from torch._inductor.codegen.wrapper import (  # 从代码生成包装器模块导入所需类
        EnterSubgraphLine,  # 进入子图的代码行标记
        ExitSubgraphLine,  # 退出子图的代码行标记
        MemoryPlanningLine,  # 内存规划代码行标记
        MemoryPlanningState,  # 内存规划状态管理类
        SubgraphPythonWrapperCodegen,  # 子图 Python 包装器代码生成器
    )
    from torch._inductor.virtualized import V  # 导入虚拟化上下文管理器

    def get_output_names(graph_outputs) -> list[str]:
        """获取图输出节点的名称列表。

        遍历图输出节点，为不同类型的节点生成对应的名称。
        """
        import itertools  # 导入迭代工具模块

        names = []  # 初始化名称列表
        shape_counter = itertools.count(0)  # 创建形状节点计数器，从 0 开始
        none_counter = itertools.count(0)  # 创建空值节点计数器，从 0 开始
        for node in graph_outputs:  # 遍历所有图输出节点
            if isinstance(node, ir.NoneAsConstantBuffer):  # 如果节点是 None 常量缓冲区
                names.append(f"{V.graph.name}_none{next(none_counter)}")  # 生成 none 类型的名称
            elif isinstance(node, ir.ShapeAsConstantBuffer):  # 如果节点是形状常量缓冲区
                names.append(f"{V.graph.name}_shape{next(shape_counter)}")  # 生成 shape 类型的名称
            else:  # 其他类型的节点
                names.append(node.get_name())  # 直接获取节点名称
        return names  # 返回名称列表

    if (  # 检查是否在子图包装器代码生成器中且存在分区签名
        isinstance(V.graph.wrapper_code, SubgraphPythonWrapperCodegen)  # 判断包装器代码是否为子图类型
        and V.graph.wrapper_code.partition_signatures is not None  # 判断是否存在分区签名
    ):
        out_names = get_output_names(  # 从分区签名的输出节点获取名称
            V.graph.wrapper_code.partition_signatures.output_nodes  # 访问分区签名中的输出节点
        )
    else:  # 不在子图包装器中或没有分区签名
        out_names = V.graph.get_output_names()  # 直接从图中获取输出名称

    while (  # 循环移除尾部无用的内存规划行
        self.lines  # 确保行列表不为空
        and isinstance(self.lines[-1], MemoryPlanningLine)  # 确保最后一行是内存规划行
        and self.lines[-1].node.name not in out_names  # type: ignore[attr-defined]  # 确保该行的节点不在输出名称中
    ):
        # these lines will be pointless
        self.lines.pop()  # 移除无用的内存规划行

    # codegen allocations in two passes
    planning_states = [MemoryPlanningState()]  # 初始化内存规划状态栈，包含一个初始状态
    past_planning_states = []  # 初始化已完成的规划状态列表
    for i in range(len(self.lines)):  # 遍历所有代码行
        line = self.lines[i]  # 获取当前行
        if isinstance(line, MemoryPlanningLine):  # 如果是内存规划行
            self.lines[i] = line.plan(planning_states[-1])  # 使用当前规划状态执行内存规划
        elif isinstance(line, EnterSubgraphLine):  # 如果是进入子图标记
            planning_states.append(MemoryPlanningState())  # 为子图创建新的规划状态并入栈
        elif isinstance(line, ExitSubgraphLine):  # 如果是退出子图标记
            past_planning_states.append(planning_states.pop())  # 将子图的规划状态出栈并保存
    past_planning_states.append(planning_states.pop())  # 将最后一个规划状态出栈并保存
    assert len(planning_states) == 0  # 断言规划状态栈已清空，确保子图嵌套正确


# ===================================================
# torch 2.9 Inductor get_graph_partition_signature monkeypatch
# ===================================================
# This change monkeypatches get_graph_partition_signature in pytorch 2.9.0 to
# fix inductor partition + attention-nvfp4 quant fusion, tested in
# `tests/compile/test_fusion_attn.py::test_attn_quant`.
# For more context, see https://github.com/pytorch/pytorch/pull/165815.


def get_graph_partition_signature_patched(
    self, partitions, skip_cudagraphs: list[bool]
):
    """修补后的图分区签名获取函数。

    对 PyTorch 2.9.0 的 get_graph_partition_signature 进行猴子补丁，
    修复 inductor 分区与 attention-nvfp4 量化融合的问题。
    测试用例：tests/compile/test_fusion_attn.py::test_attn_quant。
    详情参见 https://github.com/pytorch/pytorch/pull/165815。
    """
    from torch._inductor import dependencies  # 导入 Inductor 依赖分析模块
    from torch._inductor.ir import GraphPartitionSignature, MutationOutput, NoneLayout  # 导入图分区签名、变异输出和空布局类
    from torch._inductor.virtualized import V  # 导入虚拟化上下文管理器
    from torch.utils._ordered_set import OrderedSet  # 导入有序集合类

    signatures = []  # 初始化分区签名列表

    unmet_output_names = OrderedSet(V.graph.get_output_names())  # 获取图的输出名称，初始化为未满足的输出集合
    name_to_node = self.get_name_to_nodes()  # 获取名称到节点的映射字典

    def is_none_layout(buf_name: str) -> bool:
        """检查缓冲区是否为 NoneLayout 布局。

        NoneLayout 布局的缓冲区不会被分配内存，
        因此图分区不应将其作为输入或输出。
        """
        buf = self.name_to_buf.get(buf_name, None)  # 根据名称查找缓冲区

        if buf is None:  # 如果缓冲区不存在
            return False  # 返回 False

        if isinstance(buf.node.layout, NoneLayout):  # 如果缓冲区布局为 NoneLayout
            if isinstance(buf.node, MutationOutput) and (  # 如果是变异输出节点
                real_name := self.mutation_real_name.get(buf_name, None)  # 获取变异操作的真实名称
            ):
                return is_none_layout(real_name)  # 递归检查真实名称对应的缓冲区

            return True  # 确认是 NoneLayout

        return False  # 不是 NoneLayout

    for partition, skip_cudagraph in zip(  # 逆序遍历所有分区及其对应的 cudagraph 跳过标志
        reversed(partitions), reversed(skip_cudagraphs)  # 反转分区和跳过标志列表
    ):
        output_names: OrderedSet[str] = OrderedSet()  # 初始化当前分区的输出名称集合

        for node in partition:  # 遍历当前分区中的所有节点
            output_names.update(node.outputs_by_name.keys())  # 收集节点的所有输出名称

        returned_output_names = output_names.intersection(unmet_output_names)  # 计算当前分区输出与未满足输出的交集

        # all reads/writes are partition inputs except those generated
        # within the partition and tensor constants
        read_writes = dependencies.ReadWrites.merge_list(  # 合并当前分区所有节点的读写依赖
            [node.read_writes for node in partition]  # 收集每个节点的读写信息
        )

        # WeakDep is fake dependency on unused buffer. It should not appear
        # in partition_input_names for inputs that are actually read or written.
        partition_input_names = (  # 计算分区的输入名称集合
            OrderedSet(  # 创建有序集合
                [
                    x.name  # 获取依赖项的名称
                    for x in read_writes.reads | read_writes.writes  # 遍历所有读写依赖
                    if not is_none_layout(x.name)  # 排除 NoneLayout 布局的缓冲区
                ]
            )
            - output_names  # 从读写名称中去除本分区的输出名称（因为它们是在分区内部生成的）
        )

        partition_input_names = OrderedSet(  # 将变异名称映射为真实名称
            self.mutation_real_name.get(name, name) for name in partition_input_names  # 对每个输入名称查找真实名称
        )

        buffer_names_to_free: OrderedSet[str] = OrderedSet()  # 初始化需要释放的缓冲区名称集合
        for node in partition:  # 遍历分区中的节点
            buffer_names_to_free.update(node.last_usage)  # 收集节点最后一次使用的缓冲区名称

        # buffer_names_to_free may contain buffers allocated in previous
        # graph partitions. These buffers should also be a partition
        # input.
        extra_input_names = [  # 找出需要释放但不是本分区输出的缓冲区，它们是来自前序分区的输入
            name  # 缓冲区名称
            for name in (buffer_names_to_free - output_names)  # 从需要释放的名称中去除本分区输出
            if name in name_to_node  # 仅保留存在于名称映射中的缓冲区
        ]
        partition_input_names.update(extra_input_names)  # 将额外输入名称加入分区输入集合

        input_nodes = {  # 构建输入名称到节点的映射字典
            name: name_to_node[name]  # 名称到节点的映射
            for name in partition_input_names  # 遍历分区输入名称
            if name in name_to_node  # 仅包含存在映射关系的名称
        }
        input_deallocation = {  # 构建输入名称到是否需要释放的映射字典
            name: name in buffer_names_to_free  # 判断该输入是否在需要释放的列表中
            for name in partition_input_names  # 遍历分区输入名称
            if name in name_to_node  # 仅包含存在映射关系的名称
        }

        # if an input tensor is not freed in the partition function, it should
        # also be returned as an output. This brings benefits to cudagraph
        # since the returned output tensor is a cudagraph managed tensor with
        # a static tensor address.
        extra_output_names = [  # 找出未在本分区释放的输入张量，它们也应作为输出返回
            name  # 张量名称
            for name in partition_input_names  # 遍历分区输入名称
            if name in name_to_node and name not in buffer_names_to_free  # 存在于节点映射中且不需要释放
        ]

        returned_output_names.update(extra_output_names)  # 将额外输出名称加入返回输出集合

        returned_output_names = OrderedSet(  # 将返回输出名称映射为真实名称
            self.mutation_real_name.get(name, name) for name in returned_output_names  # 对每个名称查找真实名称
        )

        output_nodes = [  # 构建输出节点列表
            name_to_node[name]  # 根据名称获取节点
            for name in returned_output_names  # 遍历返回输出名称
            if not is_none_layout(name)  # 排除 NoneLayout 布局的节点
        ]

        constant_names = [  # 找出分区输入中属于常量的名称
            name for name in partition_input_names if name in V.graph.constants  # 检查名称是否在图常量中
        ]

        symbol_inputs = self.get_graph_partition_symbol_inputs(partition, input_nodes)  # 获取分区的符号输入（动态形状相关）

        partition_signature = GraphPartitionSignature(  # 创建图分区签名对象
            symbol_inputs,  # 符号输入
            input_nodes,  # 输入节点映射
            output_nodes,  # 输出节点列表
            input_deallocation,  # 输入释放标记映射
            skip_cudagraph,  # 是否跳过 CUDAGraph
            constant_names,  # 常量名称列表
        )

        signatures.append(partition_signature)  # 将分区签名添加到列表

        unmet_output_names = partition_input_names.union(  # 更新未满足的输出名称集合
            unmet_output_names - returned_output_names  # 去除已满足的输出，加入新的输入需求
        )

    return signatures[::-1]  # 反转签名列表并返回（因为之前是逆序遍历的）


# ========================================
# torch 2.9 Inductor Scheduler monkeypatch
# ========================================
# This change monkeypatches a function in Inductor to work around the following
# bug: https://github.com/vllm-project/vllm/issues/26678
#
# The bug occurs when `use_inductor_graph_partition` is turned on and there
# exists operators inside of `splitting_ops` that have an in-place mutation. In
# vllm, this specifically occurs on the operator
# vllm.unified_attention_with_output. In this case, inductor does not populate
# the inductor IR's `origin_node` field, causing an assertion error when trying
# to access the node's `origin_node` field.
#
# So, we will monkeypatch torch._inductor.scheduler.Scheduler.should_partition
# so that it does not access the inductor IR node's `origin_node` field and just
# returns True if a node is registered as having a custom partition function.
# This is ok for now since vllm's implementation of the custom partition
# functions just return True.
# ========================================


def should_partition_patched(self, node, should_log: bool = False) -> bool:
    """修补后的分区判断函数。

    对 torch._inductor.scheduler.Scheduler.should_partition 进行猴子补丁，
    避免访问 inductor IR 节点的 origin_node 字段（该字段在某些情况下未被填充）。
    当节点注册了自定义分区函数时直接返回 True。
    详情参见 https://github.com/vllm-project/vllm/issues/26678。
    """
    # This is a patched version of
    # torch._inductor.scheduler.Scheduler.should_partition that modifies
    # the following piece of code so that we always return True:
    # https://github.com/pytorch/pytorch/blob/ecb53078faf86ca1b33277df33b82985675bb011/torch/_inductor/scheduler.py#L4712-L4724

    import torch._inductor.ir as ir  # 导入 Inductor 中间表示模块
    from torch._inductor.scheduler import (  # 从调度器模块导入所需类
        BaseSchedulerNode,  # 基础调度器节点类
        FusedSchedulerNode,  # 融合调度器节点类
    )
    from torch._inductor.utils import (  # 从 Inductor 工具模块导入所需函数
        _unstable_customized_partition_wrapper,  # 自定义分区包装器
        is_cudagraph_unsafe_op,  # 判断操作是否对 CUDAGraph 不安全的函数
        maybe_log_cudagraph_partition,  # 可能记录 CUDAGraph 分区日志的函数
    )

    # Allow users to manually specify if a node should be partitioned
    # Can only do this for FallbackKernels
    ir_node = node.node  # 获取调度器节点对应的 IR 节点
    if isinstance(ir_node, torch._inductor.ir.FallbackKernel) and (  # 如果是回退内核节点
        op := ir_node.op_overload  # 获取操作重载
    ):
        op_overload_packet_name = op.name()  # 获取操作重载包的名称
        op_overload_name = (  # 构造完整的操作重载名称
            f"{op_overload_packet_name}.{op._overloadname}"  # 包含包名和重载名
            if isinstance(op, torch._ops.OpOverload)  # 如果是 OpOverload 类型
            else op_overload_packet_name  # 否则直接使用包名
        )
        if (  # 检查操作是否在自定义分区操作列表中
            op_overload_packet_name  # 检查操作包名
            in torch._inductor.config.custom_should_partition_ops  # 是否在自定义分区操作配置中
            or op_overload_name in torch._inductor.config.custom_should_partition_ops  # 或者完整重载名在配置中
        ):
            assert isinstance(op, torch._ops.OpOverload)  # 断言操作为 OpOverload 类型
            return True  # 在自定义分区列表中，直接返回 True 表示需要分区

    # When not using cudagraphs, keep all kernels in the `call` function
    # instead of graph partition functions, since graph partition only brings
    # benefit to cudagraph
    if (  # 检查是否未使用 CUDAGraph 且没有自定义分区包装器
        not torch._inductor.config.triton.cudagraphs  # 未启用 CUDAGraph
        and _unstable_customized_partition_wrapper.wrapper is None  # 没有自定义分区包装器
    ):
        return True  # 不需要分区，所有内核保留在 call 函数中

    # avoid duplicating logs when should_partition is called multiple times
    # on the same node
    def noop_log(msg: str, node: BaseSchedulerNode | None) -> None:
        """空操作日志函数，避免重复记录日志。"""
        return  # 不执行任何操作

    log_partition_reason = maybe_log_cudagraph_partition if should_log else noop_log  # 根据 should_log 选择日志函数

    if isinstance(node, FusedSchedulerNode):  # 如果是融合调度器节点
        return any(self.should_partition(snode) for snode in node.snodes)  # 递归检查子节点是否需要分区

    assert node.node is not None  # 断言节点的 IR 节点不为空

    if not node.is_gpu():  # 如果节点不在 GPU 上运行
        log_partition_reason("non gpu ops", node=node)  # 记录分区原因：非 GPU 操作

        return True  # 非 GPU 操作需要分区

    if isinstance(node.node, ir.DeviceCopy):  # 如果是设备间拷贝操作
        log_partition_reason("DeviceCopy ops", node=node)  # 记录分区原因：设备拷贝操作
        return True  # 设备拷贝需要分区

    if isinstance(node.node, ir.Conditional):  # 如果是条件操作
        log_partition_reason("Conditional ops", node=node)  # 记录分区原因：条件操作
        return True  # 条件操作需要分区

    if getattr(node.node, "unbacked_bindings", None):  # 如果节点有未支持的绑定
        log_partition_reason("unbacked binding ops", node=node)  # 记录分区原因：未支持的绑定操作
        return True  # 有未支持绑定的操作需要分区

    if is_cudagraph_unsafe_op(node.node):  # 如果操作对 CUDAGraph 不安全
        log_partition_reason("CUDAGraph-unsafe custom ops", node=node)  # 记录分区原因：CUDAGraph 不安全的自定义操作
        return True  # CUDAGraph 不安全的操作需要分区

    return False  # 以上条件均不满足，不需要分区


def _update_scheduler_patched(self) -> None:
    """修补后的调度器更新函数。

    从 torch._inductor.graph.GraphLowering._update_scheduler 复制并修补，
    在初始化调度器时注入 should_partition 和 get_graph_partition_signature 的补丁。
    初始化调度器时不应生成 CUBIN 文件，以避免影响基准测试和融合决策。
    """
    # Copied from torch._inductor.graph.GrahLowering._update_scheduler. Patches
    # this method so that we can patch Scheduler.should_partition with the
    # function above
    import torch._inductor.config as config  # 导入 Inductor 配置模块
    from torch._inductor.scheduler import Scheduler  # 导入调度器类

    Scheduler.should_partition = should_partition_patched  # 用修补版本替换 should_partition 方法
    Scheduler.get_graph_partition_signature = get_graph_partition_signature_patched  # 用修补版本替换 get_graph_partition_signature 方法

    with config.patch("triton.store_cubin", False):  # 临时禁用 CUBIN 存储，避免影响基准测试
        self.scheduler = Scheduler(self.operations)  # 使用修补后的调度器类创建调度器实例


# ===================================================
# torch 2.9 Inductor get_raw_stream workaround
# ===================================================
# Workaround for TorchInductor autotune using get_raw_stream() without defining it.
# This occurs when compile_sizes > 1 in compilation_config.
# For more context, see https://github.com/vllm-project/vllm/issues/30905.
def _patch_get_raw_stream_if_needed():
    """修补 TorchInductor 自动调优中 get_raw_stream() 未定义的问题。

    当 compilation_config 中 compile_sizes > 1 时，TorchInductor 的自动调优
    会使用 get_raw_stream() 但未预先定义该函数。
    此补丁将 get_raw_stream 注入到 builtins 中以解决该问题。
    详情参见 https://github.com/vllm-project/vllm/issues/30905。
    """
    from vllm.utils.torch_utils import is_torch_equal  # 导入 PyTorch 版本比较工具函数

    # Only apply the patch for torch 2.9.0 or 2.9.1
    if is_torch_equal("2.9.0") or is_torch_equal("2.9.1"):  # 仅对 torch 2.9.0 或 2.9.1 版本应用补丁
        import builtins  # 导入内建模块

        # Check if CUDA functionality is available without initializing CUDA
        # _cuda_getCurrentRawStream only exists in CUDA builds of PyTorch
        if hasattr(torch._C, "_cuda_getCurrentRawStream"):  # 检查 PyTorch 是否包含 CUDA 原始流获取函数
            from torch._C import _cuda_getCurrentRawStream as _get_raw_stream  # 导入 CUDA 原始流获取函数

            builtins.get_raw_stream = _get_raw_stream  # type: ignore[attr-defined]  # 将函数注入到 builtins 中供全局访问


_patch_get_raw_stream_if_needed()  # 调用补丁函数，修复 get_raw_stream 未定义问题

if is_torch_equal("2.9.0"):  # 如果 PyTorch 版本为 2.9.0
    from torch._inductor.codegen.wrapper import PythonWrapperCodegen  # 导入 Python 包装器代码生成器类
    from torch._inductor.graph import GraphLowering  # 导入图降低类
    from torch.utils._config_module import _Config, _ConfigEntry  # 导入配置模块的配置类和配置条目类

    # `custom_should_partition_ops` is a new config after 2.9.0. So this would
    # not overwrite any user configs.
    torch._inductor.config._config["custom_should_partition_ops"] = _ConfigEntry(  # 注册新的配置项：自定义分区操作列表
        _Config(default=[])  # 默认值为空列表
    )

    PythonWrapperCodegen.memory_plan_reuse = memory_plan_reuse_patched  # 用修补版本替换 memory_plan_reuse 方法
    GraphLowering._update_scheduler = _update_scheduler_patched  # 用修补版本替换 _update_scheduler 方法
