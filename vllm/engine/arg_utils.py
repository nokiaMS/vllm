# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 以上为 Apache-2.0 开源许可证声明和版权声明

import argparse  # 导入命令行参数解析模块
import copy  # 导入深拷贝模块，用于对象的深复制
import dataclasses  # 导入数据类模块，用于数据类操作
import functools  # 导入函数工具模块，提供高阶函数如 lru_cache 缓存装饰器
import json  # 导入 JSON 模块，用于 JSON 数据的解析和序列化
import sys  # 导入系统模块，用于访问命令行参数等系统功能
from collections.abc import Callable  # 从集合抽象基类导入 Callable 类型，用于表示可调用对象
from dataclasses import MISSING, dataclass, fields, is_dataclass  # 从数据类模块导入缺失值标记、数据类装饰器、字段获取函数和数据类检查函数
from itertools import permutations  # 从迭代工具模块导入排列组合函数
from types import UnionType  # 从类型模块导入 UnionType，用于处理 X | Y 形式的联合类型
from typing import (  # 从类型提示模块导入以下类型
    TYPE_CHECKING,  # 类型检查标志，仅在类型检查工具运行时为 True
    Annotated,  # 带注解的类型提示
    Any,  # 任意类型
    Literal,  # 字面量类型，用于限定具体值
    TypeAlias,  # 类型别名声明
    TypeVar,  # 类型变量，用于泛型编程
    Union,  # 联合类型 Union[X, Y]
    cast,  # 类型强制转换（仅用于类型检查，运行时无操作）
    get_args,  # 获取泛型类型的参数
    get_origin,  # 获取泛型类型的原始类型
)

import huggingface_hub  # 导入 HuggingFace Hub 库，用于模型下载和管理
import regex as re  # 导入正则表达式库（regex 版本，功能比标准 re 更强大）
import torch  # 导入 PyTorch 深度学习框架
from pydantic import TypeAdapter, ValidationError  # 从 Pydantic 导入类型适配器和验证错误类，用于数据验证
from pydantic.fields import FieldInfo  # 从 Pydantic 导入字段信息类
from typing_extensions import TypeIs  # 从 typing_extensions 导入 TypeIs，用于类型缩小

import vllm.envs as envs  # 导入 vLLM 环境变量模块
# 从 vLLM 配置模块导入各种配置类
from vllm.config import (
    AttentionConfig,  # 注意力机制配置
    CacheConfig,  # KV 缓存配置
    CompilationConfig,  # 编译优化配置（如 CUDAGraph）
    ConfigType,  # 配置类型别名
    DeviceConfig,  # 设备配置（CPU/GPU等）
    ECTransferConfig,  # EC（嵌入式连接器）传输配置
    EPLBConfig,  # 专家并行负载均衡配置
    KernelConfig,  # 内核配置
    KVEventsConfig,  # KV 事件配置
    KVTransferConfig,  # KV 传输配置（用于分离式推理）
    LoadConfig,  # 模型加载配置
    LoRAConfig,  # LoRA 适配器配置
    ModelConfig,  # 模型配置
    MultiModalConfig,  # 多模态配置
    ObservabilityConfig,  # 可观测性配置（指标、追踪等）
    OffloadConfig,  # 卸载配置（模型权重卸载到 CPU）
    ParallelConfig,  # 并行配置（张量并行、流水线并行等）
    PoolerConfig,  # 池化层配置
    PrefetchOffloadConfig,  # 预取卸载配置
    ProfilerConfig,  # 性能分析器配置
    SchedulerConfig,  # 调度器配置
    SpeculativeConfig,  # 推测解码配置
    StructuredOutputsConfig,  # 结构化输出配置
    UVAOffloadConfig,  # UVA（统一虚拟地址）卸载配置
    VllmConfig,  # vLLM 总配置类
    WeightTransferConfig,  # 权重传输配置
    get_attr_docs,  # 获取属性文档字符串的辅助函数
)
# 从缓存配置模块导入缓存相关的类型定义
from vllm.config.cache import (
    CacheDType,  # 缓存数据类型枚举
    KVOffloadingBackend,  # KV 缓存卸载后端类型
    MambaCacheMode,  # Mamba 模型缓存模式
    MambaDType,  # Mamba 模型缓存数据类型
    PrefixCachingHashAlgo,  # 前缀缓存哈希算法
)
from vllm.config.device import Device  # 导入设备类型定义
from vllm.config.kernel import MoEBackend  # 导入 MoE（混合专家）后端类型
from vllm.config.lora import MaxLoRARanks  # 导入 LoRA 最大秩类型
from vllm.config.model import (  # 从模型配置导入各种选项类型
    ConvertOption,  # 模型转换选项
    HfOverrides,  # HuggingFace 配置覆盖
    LogprobsMode,  # 对数概率计算模式
    ModelDType,  # 模型数据类型
    RunnerOption,  # 运行器选项（generate/pooling等）
    TokenizerMode,  # 分词器模式
)
from vllm.config.multimodal import MMCacheType, MMEncoderTPMode  # 导入多模态缓存类型和编码器张量并行模式
from vllm.config.observability import DetailedTraceModules  # 导入详细追踪模块类型
from vllm.config.parallel import (  # 从并行配置导入并行相关类型
    All2AllBackend,  # All-to-All 通信后端
    DataParallelBackend,  # 数据并行后端
    DCPCommBackend,  # 解码上下文并行通信后端
    DistributedExecutorBackend,  # 分布式执行器后端
    ExpertPlacementStrategy,  # 专家放置策略
)
from vllm.config.scheduler import SchedulerPolicy  # 导入调度策略类型
from vllm.config.utils import get_field  # 导入获取数据类字段默认值的辅助函数
from vllm.config.vllm import OptimizationLevel, PerformanceMode  # 导入优化级别和性能模式类型
from vllm.logger import init_logger, suppress_logging  # 导入日志初始化函数和日志抑制上下文管理器
from vllm.platforms import CpuArchEnum, current_platform  # 导入 CPU 架构枚举和当前平台对象
from vllm.plugins import load_general_plugins  # 导入加载通用插件的函数
from vllm.ray.lazy_utils import is_in_ray_actor, is_ray_initialized  # 导入 Ray 环境检测工具函数
from vllm.transformers_utils.config import (  # 从 transformers 配置工具导入
    is_interleaved,  # 检查模型是否为交错滑动窗口架构
    maybe_override_with_speculators,  # 检测并处理推测器模型覆盖
)
from vllm.transformers_utils.gguf_utils import is_gguf  # 导入 GGUF 格式检测函数
from vllm.transformers_utils.repo_utils import get_model_path  # 导入获取模型本地路径的函数
from vllm.transformers_utils.utils import is_cloud_storage  # 导入云存储路径检测函数
from vllm.utils.argparse_utils import FlexibleArgumentParser  # 导入灵活的命令行参数解析器
from vllm.utils.mem_constants import GiB_bytes  # 导入 GiB 字节数常量
from vllm.utils.network_utils import get_ip  # 导入获取本机 IP 地址的函数
from vllm.utils.torch_utils import resolve_kv_cache_dtype_string  # 导入解析 KV 缓存数据类型字符串的函数
from vllm.v1.attention.backends.registry import AttentionBackendEnum  # 导入注意力后端枚举类型
from vllm.v1.sample.logits_processor import LogitsProcessor  # 导入 logits 处理器基类

# 仅在类型检查时导入以下模块（避免循环导入和运行时开销）
if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods  # 量化方法类型
    from vllm.model_executor.model_loader import LoadFormats  # 模型加载格式类型
    from vllm.usage.usage_lib import UsageContext  # 使用场景上下文类型
    from vllm.v1.executor import Executor  # V1 执行器类型
else:
    # 运行时将这些类型设为 Any，避免导入依赖
    Executor = Any  # 执行器类型占位
    QuantizationMethods = Any  # 量化方法类型占位
    LoadFormats = Any  # 加载格式类型占位
    UsageContext = Any  # 使用上下文类型占位


# 初始化当前模块的日志记录器
logger = init_logger(__name__)

# object 类型用于允许特殊的类型提示形式（如 Literal、Union 等）
T = TypeVar("T")  # 定义泛型类型变量 T
TypeHint: TypeAlias = type[Any] | object  # 类型提示的类型别名，可以是具体类型或 object
TypeHintT: TypeAlias = type[T] | object  # 带泛型参数的类型提示别名


def parse_type(return_type: Callable[[str], T]) -> Callable[[str], T]:
    """将字符串解析为指定类型的工厂函数。

    参数:
        return_type: 目标类型的转换函数（如 int、float 等）
    返回:
        一个将字符串转换为目标类型的函数，转换失败时抛出 argparse 错误
    """
    def _parse_type(val: str) -> T:
        """内部解析函数，尝试将字符串值转换为目标类型"""
        try:
            return return_type(val)  # 尝试使用目标类型函数转换字符串
        except ValueError as e:
            # 转换失败时抛出 argparse 类型错误
            raise argparse.ArgumentTypeError(
                f"Value {val} cannot be converted to {return_type}."
            ) from e

    return _parse_type  # 返回解析函数


def optional_type(return_type: Callable[[str], T]) -> Callable[[str], T | None]:
    """创建一个可选类型解析器，允许值为空字符串或 "None" 时返回 None。

    参数:
        return_type: 目标类型的转换函数
    返回:
        一个将字符串转换为目标类型或 None 的函数
    """
    def _optional_type(val: str) -> T | None:
        """内部可选类型解析函数"""
        if val == "" or val == "None":  # 如果值为空或 "None"，返回 None
            return None
        return parse_type(return_type)(val)  # 否则使用 parse_type 进行类型转换

    return _optional_type  # 返回可选类型解析函数


def union_dict_and_str(val: str) -> str | dict[str, str] | None:
    """解析可能是字典（JSON）或普通字符串的值。

    参数:
        val: 输入字符串
    返回:
        如果输入看起来像 JSON 对象则解析为字典，否则返回字符串
    """
    if not re.match(r"(?s)^\s*{.*}\s*$", val):  # 如果不匹配 JSON 对象格式（花括号包围）
        return str(val)  # 作为普通字符串返回
    return optional_type(json.loads)(val)  # 尝试将其解析为 JSON 对象


def is_type(type_hint: TypeHint, type: TypeHintT) -> TypeIs[TypeHintT]:
    """检查类型提示是否为指定类型。

    参数:
        type_hint: 要检查的类型提示
        type: 目标类型
    返回:
        如果类型提示是目标类型或其原始类型是目标类型，则返回 True
    """
    return type_hint is type or get_origin(type_hint) is type  # 直接比较或通过 get_origin 比较泛型的原始类型


def contains_type(type_hints: set[TypeHint], type: TypeHintT) -> bool:
    """检查类型提示集合中是否包含指定类型。

    参数:
        type_hints: 类型提示集合
        type: 要查找的目标类型
    返回:
        如果集合中存在匹配的类型则返回 True
    """
    return any(is_type(type_hint, type) for type_hint in type_hints)  # 遍历集合检查是否有匹配的类型


def get_type(type_hints: set[TypeHint], type: TypeHintT) -> TypeHintT:
    """从类型提示集合中获取指定类型。

    参数:
        type_hints: 类型提示集合
        type: 要查找的目标类型
    返回:
        匹配的类型提示，如果未找到则返回 None
    """
    return next((th for th in type_hints if is_type(th, type)), None)  # 返回第一个匹配的类型，未找到返回 None


def literal_to_kwargs(type_hints: set[TypeHint]) -> dict[str, Any]:
    """从类型提示集合中的 Literal 类型提取 argparse 的 type 和 choices 参数。

    从 `type_hints` 中获取 `Literal` 类型提示，并将其转换为 argparse 的关键字参数。
    如果 `type_hints` 同时包含 `str` 类型，则使用 `metavar` 代替 `choices`（仅显示提示不做限制）。

    参数:
        type_hints: 类型提示集合
    返回:
        包含 type 和 choices/metavar 的字典
    """
    type_hint = get_type(type_hints, Literal)  # 从集合中获取 Literal 类型
    options = get_args(type_hint)  # 获取 Literal 的所有可选值
    option_type = type(options[0])  # 获取第一个选项的类型作为基准类型
    if not all(isinstance(option, option_type) for option in options):  # 检查所有选项是否类型一致
        raise ValueError(
            "All options must be of the same type. "
            f"Got {options} with types {[type(c) for c in options]}"
        )
    # 如果类型提示中还包含 str 类型，使用 metavar（不严格限制选项）；否则使用 choices（严格限制）
    kwarg = "metavar" if contains_type(type_hints, str) else "choices"
    return {"type": option_type, kwarg: sorted(options)}  # 返回类型和排序后的选项


def collection_to_kwargs(type_hints: set[TypeHint], type: TypeHint) -> dict[str, Any]:
    """将集合类型（list、tuple、set）的类型提示转换为 argparse 关键字参数。

    参数:
        type_hints: 类型提示集合
        type: 集合类型（list、tuple 或 set）
    返回:
        包含 type 和 nargs 的字典，用于 argparse 参数配置
    """
    type_hint = get_type(type_hints, type)  # 从集合中获取指定的集合类型
    types = get_args(type_hint)  # 获取集合元素的类型参数
    elem_type = types[0]  # 获取元素类型（取第一个类型参数）

    # 处理省略号（Ellipsis），确保所有非省略号的元素类型相同
    assert all(t is elem_type for t in types if t is not Ellipsis), (
        f"All non-Ellipsis elements must be of the same type. Got {types}."
    )

    # 处理联合类型（Union）
    if get_origin(elem_type) in {Union, UnionType}:
        # Union 用于 Union[X, Y] 形式，UnionType 用于 X | Y 形式
        assert str in get_args(elem_type), (
            "If element can have multiple types, one must be 'str' "
            f"(i.e. 'list[int | str]'). Got {elem_type}."
        )
        elem_type = str  # 如果元素类型是联合类型且包含 str，则统一使用 str

    return {
        "type": elem_type,  # 元素类型
        # nargs: 如果不是 tuple 或包含省略号，则接受一个或多个参数（"+"）；否则接受固定数量的参数
        "nargs": "+" if type is not tuple or Ellipsis in types else len(types),
    }


def is_not_builtin(type_hint: TypeHint) -> bool:
    """检查类型是否不是内置类型（如 int、str、float 等）。

    参数:
        type_hint: 要检查的类型提示
    返回:
        如果不是内置类型则返回 True
    """
    return type_hint.__module__ != "builtins"  # 通过检查模块是否为 "builtins" 来判断


def get_type_hints(type_hint: TypeHint) -> set[TypeHint]:
    """从 Annotated 或 Union 类型提示中递归提取所有基础类型。

    参数:
        type_hint: 要解析的类型提示
    返回:
        包含所有基础类型的集合
    """
    type_hints: set[TypeHint] = set()  # 初始化类型集合
    origin = get_origin(type_hint)  # 获取泛型的原始类型
    args = get_args(type_hint)  # 获取泛型的类型参数

    if origin is Annotated:  # 如果是 Annotated 类型，递归提取第一个参数的类型
        type_hints.update(get_type_hints(args[0]))
    elif origin in {Union, UnionType}:  # 如果是联合类型
        # Union 用于 Union[X, Y] 形式，UnionType 用于 X | Y 形式
        for arg in args:  # 递归提取每个联合成员的类型
            type_hints.update(get_type_hints(arg))
    else:
        type_hints.add(type_hint)  # 基础类型直接加入集合

    return type_hints  # 返回所有提取的类型


# 检测是否需要生成帮助文本（命令行包含 --help 或正在生成文档）
NEEDS_HELP = (
    any("--help" in arg for arg in sys.argv)  # 检查命令行是否包含 --help 参数
    or (argv0 := sys.argv[0]).endswith("mkdocs")  # 检查是否通过 mkdocs 命令运行
    or argv0.endswith("mkdocs/__main__.py")  # 检查是否通过 python -m mkdocs 运行
)


@functools.lru_cache(maxsize=30)  # 使用 LRU 缓存装饰器，最多缓存 30 个不同配置类的结果
def _compute_kwargs(cls: ConfigType) -> dict[str, dict[str, Any]]:
    """计算给定配置数据类的 argparse 关键字参数（内部缓存版本）。

    遍历配置类的所有字段，根据字段的类型提示和默认值，
    生成对应的 argparse add_argument 关键字参数字典。

    参数:
        cls: 配置数据类类型
    返回:
        字段名到 argparse 关键字参数字典的映射
    """
    # 仅在需要生成帮助文本时才获取属性文档，以节省时间
    cls_docs = get_attr_docs(cls) if NEEDS_HELP else {}
    kwargs = {}  # 存储所有字段的 argparse 参数字典
    for field in fields(cls):  # 遍历数据类的每个字段
        # 获取字段所有可能的类型集合
        type_hints: set[TypeHint] = get_type_hints(field.type)

        # 检查字段类型中是否有数据类类型，如果有可以使用 JSON 验证
        generator = (th for th in type_hints if is_dataclass(th))
        dataclass_cls = next(generator, None)  # 获取第一个数据类类型，如果没有则为 None

        # 获取字段的默认值
        if field.default is not MISSING:  # 如果字段有直接默认值
            default = field.default
            # 处理 pydantic.Field 的默认值
            if isinstance(default, FieldInfo):  # 如果默认值是 Pydantic 的 FieldInfo 对象
                if default.default_factory is None:  # 如果没有默认工厂函数
                    default = default.default  # 直接使用默认值
                else:
                    # VllmConfig 的 Field 将 default_factory 设置为配置类
                    # 这些类在初始化时可能会输出日志，会造成混淆，所以抑制日志
                    with suppress_logging():
                        default = default.default_factory()  # type: ignore[call-arg]
        elif field.default_factory is not MISSING:  # 如果字段有默认工厂函数
            default = field.default_factory()  # 调用工厂函数获取默认值

        # 获取字段的帮助文本
        name = field.name  # 字段名称
        help = cls_docs.get(name, "").strip()  # 从文档中获取帮助文本并去除首尾空白
        # 转义 % 符号，因为 argparse 会将 % 解释为格式化字符
        help = help.replace("%", "%%")

        # 初始化该字段的 kwargs 字典，包含默认值和帮助文本
        kwargs[name] = {"default": default, "help": help}

        # 根据类型提示设置其他 argparse 参数
        json_tip = (  # JSON 格式提示文本
            "Should either be a valid JSON string or JSON keys passed individually."
        )
        if dataclass_cls is not None:  # 如果字段类型是数据类

            def parse_dataclass(val: str, cls=dataclass_cls) -> Any:
                """将 JSON 字符串解析为数据类实例"""
                try:
                    return TypeAdapter(cls).validate_json(val)  # 使用 Pydantic 验证 JSON
                except ValidationError as e:
                    raise argparse.ArgumentTypeError(repr(e)) from e  # 验证失败时抛出错误

            kwargs[name]["type"] = parse_dataclass  # 设置类型解析函数
            kwargs[name]["help"] += f"\n\n{json_tip}"  # 添加 JSON 格式提示
        elif contains_type(type_hints, bool):  # 如果字段类型包含 bool
            # 创建 --no-<name> 和 --<name> 两个标志参数
            kwargs[name]["action"] = argparse.BooleanOptionalAction
        elif contains_type(type_hints, Literal):  # 如果字段类型包含 Literal（字面量类型）
            kwargs[name].update(literal_to_kwargs(type_hints))  # 使用字面量转换函数设置 choices
        elif contains_type(type_hints, tuple):  # 如果字段类型包含 tuple（元组）
            kwargs[name].update(collection_to_kwargs(type_hints, tuple))  # 使用集合转换函数
        elif contains_type(type_hints, list):  # 如果字段类型包含 list（列表）
            kwargs[name].update(collection_to_kwargs(type_hints, list))  # 使用集合转换函数
        elif contains_type(type_hints, set):  # 如果字段类型包含 set（集合）
            kwargs[name].update(collection_to_kwargs(type_hints, set))  # 使用集合转换函数
        elif contains_type(type_hints, int):  # 如果字段类型包含 int（整数）
            if name == "max_model_len":  # 特殊处理 max_model_len 参数
                kwargs[name]["type"] = human_readable_int_or_auto  # 支持人类可读格式和 auto
                kwargs[name]["help"] += f"\n\n{human_readable_int_or_auto.__doc__}"  # 添加格式说明
            elif name in ("max_num_batched_tokens", "kv_cache_memory_bytes"):  # 特殊处理这些参数
                kwargs[name]["type"] = human_readable_int  # 支持人类可读格式（如 1k、2M）
                kwargs[name]["help"] += f"\n\n{human_readable_int.__doc__}"  # 添加格式说明
            else:
                kwargs[name]["type"] = int  # 其他整数参数使用标准 int 类型
        elif contains_type(type_hints, float):  # 如果字段类型包含 float（浮点数）
            kwargs[name]["type"] = float  # 设置为 float 类型
        elif contains_type(type_hints, dict) and (  # 如果字段类型包含 dict 且同时包含 str 或非内置类型
            contains_type(type_hints, str)
            or any(is_not_builtin(th) for th in type_hints)
        ):
            kwargs[name]["type"] = union_dict_and_str  # 使用字典和字符串联合解析器
        elif contains_type(type_hints, dict):  # 如果字段类型仅包含 dict
            kwargs[name]["type"] = parse_type(json.loads)  # 使用 JSON 解析
            kwargs[name]["help"] += f"\n\n{json_tip}"  # 添加 JSON 格式提示
        elif contains_type(type_hints, str) or any(  # 如果字段类型包含 str 或非内置类型
            is_not_builtin(th) for th in type_hints
        ):
            kwargs[name]["type"] = str  # 设置为 str 类型
        else:
            # 不支持的类型，抛出错误
            raise ValueError(f"Unsupported type {type_hints} for argument {name}.")

        # 如果类型提示是字面量序列，使用辅助函数更新 type 和 choices
        if get_origin(kwargs[name].get("type")) is Literal:
            kwargs[name].update(literal_to_kwargs({kwargs[name]["type"]}))

        # 如果类型提示中包含 None，则将参数设为可选
        # 但不处理 bool 类型，因为 argparse 对 bool 有更好的处理方式
        if type(None) in type_hints and not contains_type(type_hints, bool):
            kwargs[name]["type"] = optional_type(kwargs[name]["type"])  # 包装为可选类型解析器
            if kwargs[name].get("choices"):  # 如果有 choices 列表
                kwargs[name]["choices"].append("None")  # 添加 "None" 选项
    return kwargs  # 返回所有字段的 argparse 参数字典


def get_kwargs(cls: ConfigType) -> dict[str, dict[str, Any]]:
    """返回给定配置数据类的 argparse 关键字参数。

    如果命令行中没有 `--help` 或 `mkdocs`，则不会在帮助输出中包含属性文档。

    繁重的计算通过 functools.lru_cache 缓存，返回深拷贝副本，
    以便调用者可以修改字典而不影响缓存版本。

    参数:
        cls: 配置数据类类型
    返回:
        字段名到 argparse 关键字参数字典的深拷贝映射
    """
    return copy.deepcopy(_compute_kwargs(cls))  # 返回缓存结果的深拷贝


@dataclass  # 数据类装饰器，自动生成 __init__、__repr__ 等方法
class EngineArgs:
    """vLLM 引擎参数类。

    该数据类定义了 vLLM 推理引擎的所有可配置参数，
    包括模型配置、缓存配置、并行配置、调度配置等。
    这些参数可以通过命令行参数或编程方式设置。
    """

    # ===== 模型相关参数 =====
    model: str = ModelConfig.model  # 模型名称或路径（HuggingFace 模型 ID 或本地路径）
    enable_return_routed_experts: bool = ModelConfig.enable_return_routed_experts  # 是否启用返回路由专家信息
    model_weights: str = ModelConfig.model_weights  # 模型权重路径（可与模型配置路径不同）
    served_model_name: str | list[str] | None = ModelConfig.served_model_name  # API 中使用的模型名称
    tokenizer: str | None = ModelConfig.tokenizer  # 分词器名称或路径（默认与模型相同）
    hf_config_path: str | None = ModelConfig.hf_config_path  # HuggingFace 配置文件路径
    runner: RunnerOption = ModelConfig.runner  # 运行器类型（generate/pooling 等）
    convert: ConvertOption = ModelConfig.convert  # 模型转换选项
    skip_tokenizer_init: bool = ModelConfig.skip_tokenizer_init  # 是否跳过分词器初始化
    enable_prompt_embeds: bool = ModelConfig.enable_prompt_embeds  # 是否启用提示嵌入
    tokenizer_mode: TokenizerMode | str = ModelConfig.tokenizer_mode  # 分词器模式（auto/slow/mistral 等）
    trust_remote_code: bool = ModelConfig.trust_remote_code  # 是否信任远程代码（HuggingFace 模型可能包含自定义代码）
    allowed_local_media_path: str = ModelConfig.allowed_local_media_path  # 允许的本地媒体文件路径
    allowed_media_domains: list[str] | None = ModelConfig.allowed_media_domains  # 允许的媒体域名列表
    download_dir: str | None = LoadConfig.download_dir  # 模型下载目录
    safetensors_load_strategy: str = LoadConfig.safetensors_load_strategy  # safetensors 加载策略
    load_format: str | LoadFormats = LoadConfig.load_format  # 模型加载格式（auto/safetensors/gguf 等）
    config_format: str = ModelConfig.config_format  # 模型配置格式
    dtype: ModelDType = ModelConfig.dtype  # 模型数据类型（auto/float16/bfloat16/float32 等）
    kv_cache_dtype: CacheDType = CacheConfig.cache_dtype  # KV 缓存数据类型
    seed: int = ModelConfig.seed  # 随机种子，用于可重复性
    max_model_len: int = ModelConfig.max_model_len  # 最大模型上下文长度（token 数）
    cudagraph_capture_sizes: list[int] | None = (  # CUDAGraph 捕获的批大小列表
        CompilationConfig.cudagraph_capture_sizes
    )
    max_cudagraph_capture_size: int | None = get_field(  # CUDAGraph 捕获的最大批大小
        CompilationConfig, "max_cudagraph_capture_size"
    )
    # 注意：通过传递类来指定自定义执行器后端仅供专家使用。
    # API 可能会在不通知的情况下更改。
    distributed_executor_backend: (  # 分布式执行器后端（ray/mp/external_launcher 等）
        str | DistributedExecutorBackend | type[Executor] | None
    ) = ParallelConfig.distributed_executor_backend
    # P/D 分离（或其他分离式）工作器的数量
    # ===== 并行相关参数 =====
    pipeline_parallel_size: int = ParallelConfig.pipeline_parallel_size  # 流水线并行大小（模型层分割的 GPU 数量）
    master_addr: str = ParallelConfig.master_addr  # 分布式训练主节点地址
    master_port: int = ParallelConfig.master_port  # 分布式训练主节点端口
    nnodes: int = ParallelConfig.nnodes  # 分布式集群节点数量
    node_rank: int = ParallelConfig.node_rank  # 当前节点在集群中的排名
    distributed_timeout_seconds: int | None = ParallelConfig.distributed_timeout_seconds  # 分布式通信超时时间（秒）
    tensor_parallel_size: int = ParallelConfig.tensor_parallel_size  # 张量并行大小（模型张量分割的 GPU 数量）
    prefill_context_parallel_size: int = ParallelConfig.prefill_context_parallel_size  # 预填充阶段的上下文并行大小
    decode_context_parallel_size: int = ParallelConfig.decode_context_parallel_size  # 解码阶段的上下文并行大小
    dcp_comm_backend: DCPCommBackend = ParallelConfig.dcp_comm_backend  # 解码上下文并行通信后端
    dcp_kv_cache_interleave_size: int = ParallelConfig.dcp_kv_cache_interleave_size  # DCP KV 缓存交错大小
    cp_kv_cache_interleave_size: int = ParallelConfig.cp_kv_cache_interleave_size  # CP KV 缓存交错大小
    data_parallel_size: int = ParallelConfig.data_parallel_size  # 数据并行大小（副本数量）
    data_parallel_rank: int | None = None  # 当前数据并行副本的排名
    data_parallel_start_rank: int | None = None  # 次级节点的数据并行起始排名
    data_parallel_size_local: int | None = None  # 当前节点上运行的数据并行副本数量
    data_parallel_address: str | None = None  # 数据并行集群头节点地址
    data_parallel_rpc_port: int | None = None  # 数据并行 RPC 通信端口
    data_parallel_hybrid_lb: bool = False  # 是否启用混合负载均衡（本地+外部）
    data_parallel_external_lb: bool = False  # 是否启用外部负载均衡
    data_parallel_backend: DataParallelBackend = ParallelConfig.data_parallel_backend  # 数据并行后端（mp/ray）
    enable_expert_parallel: bool = ParallelConfig.enable_expert_parallel  # 是否启用专家并行（MoE 模型）
    moe_backend: MoEBackend = KernelConfig.moe_backend  # MoE 计算后端
    all2all_backend: All2AllBackend = ParallelConfig.all2all_backend  # All-to-All 通信后端
    enable_elastic_ep: bool = ParallelConfig.enable_elastic_ep  # 是否启用弹性专家并行
    enable_dbo: bool = ParallelConfig.enable_dbo  # 是否启用分离式批处理编排（DBO）
    ubatch_size: int = ParallelConfig.ubatch_size  # 微批次大小
    dbo_decode_token_threshold: int = ParallelConfig.dbo_decode_token_threshold  # DBO 解码 token 阈值
    dbo_prefill_token_threshold: int = ParallelConfig.dbo_prefill_token_threshold  # DBO 预填充 token 阈值
    disable_nccl_for_dp_synchronization: bool | None = (  # 是否禁用 NCCL 进行数据并行同步
        ParallelConfig.disable_nccl_for_dp_synchronization
    )
    eplb_config: EPLBConfig = get_field(ParallelConfig, "eplb_config")  # 专家并行负载均衡配置
    enable_eplb: bool = ParallelConfig.enable_eplb  # 是否启用专家并行负载均衡
    expert_placement_strategy: ExpertPlacementStrategy = (  # 专家放置策略
        ParallelConfig.expert_placement_strategy
    )
    _api_process_count: int = ParallelConfig._api_process_count  # API 进程总数（内部使用）
    _api_process_rank: int = ParallelConfig._api_process_rank  # 当前 API 进程排名（内部使用）
    max_parallel_loading_workers: int | None = (  # 最大并行模型加载工作器数量
        ParallelConfig.max_parallel_loading_workers
    )
    # ===== 缓存相关参数 =====
    block_size: int | None = None  # KV 缓存块大小（token 数），None 表示自动选择
    enable_prefix_caching: bool | None = None  # 是否启用前缀缓存，None 表示自动决定
    prefix_caching_hash_algo: PrefixCachingHashAlgo = (  # 前缀缓存哈希算法
        CacheConfig.prefix_caching_hash_algo
    )
    disable_sliding_window: bool = ModelConfig.disable_sliding_window  # 是否禁用滑动窗口注意力
    disable_cascade_attn: bool = ModelConfig.disable_cascade_attn  # 是否禁用级联注意力
    # ===== 卸载相关参数 =====
    offload_backend: str = OffloadConfig.offload_backend  # 卸载后端类型
    cpu_offload_gb: float = UVAOffloadConfig.cpu_offload_gb  # CPU 卸载的内存大小（GB）
    cpu_offload_params: set[str] = get_field(UVAOffloadConfig, "cpu_offload_params")  # 需要卸载到 CPU 的参数集合
    offload_group_size: int = PrefetchOffloadConfig.offload_group_size  # 预取卸载组大小
    offload_num_in_group: int = PrefetchOffloadConfig.offload_num_in_group  # 预取卸载组内数量
    offload_prefetch_step: int = PrefetchOffloadConfig.offload_prefetch_step  # 预取卸载步数
    offload_params: set[str] = get_field(PrefetchOffloadConfig, "offload_params")  # 需要预取卸载的参数集合
    # ===== 调度与批处理参数 =====
    gpu_memory_utilization: float = CacheConfig.gpu_memory_utilization  # GPU 内存利用率（0.0-1.0）
    kv_cache_memory_bytes: int | None = CacheConfig.kv_cache_memory_bytes  # KV 缓存内存大小（字节），None 表示自动计算
    max_num_batched_tokens: int | None = None  # 每批次最大 token 数，None 表示自动设置
    max_num_partial_prefills: int = SchedulerConfig.max_num_partial_prefills  # 并发部分预填充的最大数量
    max_long_partial_prefills: int = SchedulerConfig.max_long_partial_prefills  # 长部分预填充的最大数量
    long_prefill_token_threshold: int = SchedulerConfig.long_prefill_token_threshold  # 长预填充的 token 阈值
    max_num_seqs: int | None = None  # 每批次最大序列数，None 表示自动设置
    max_logprobs: int = ModelConfig.max_logprobs  # 返回的最大对数概率数量
    logprobs_mode: LogprobsMode = ModelConfig.logprobs_mode  # 对数概率计算模式
    disable_log_stats: bool = False  # 是否禁用统计日志记录
    aggregate_engine_logging: bool = False  # 数据并行时是否聚合引擎日志
    # ===== 模型版本与认证参数 =====
    revision: str | None = ModelConfig.revision  # 模型版本/分支/标签
    code_revision: str | None = ModelConfig.code_revision  # 代码版本（用于自定义代码模型）
    hf_token: bool | str | None = ModelConfig.hf_token  # HuggingFace 访问令牌
    hf_overrides: HfOverrides = get_field(ModelConfig, "hf_overrides")  # HuggingFace 配置覆盖
    tokenizer_revision: str | None = ModelConfig.tokenizer_revision  # 分词器版本
    # ===== 量化相关参数 =====
    quantization: QuantizationMethods | str | None = ModelConfig.quantization  # 量化方法（awq/gptq/fp8 等）
    allow_deprecated_quantization: bool = ModelConfig.allow_deprecated_quantization  # 是否允许已弃用的量化方法
    enforce_eager: bool = ModelConfig.enforce_eager  # 是否强制使用 eager 模式（禁用 CUDAGraph）
    disable_custom_all_reduce: bool = ParallelConfig.disable_custom_all_reduce  # 是否禁用自定义 AllReduce 通信
    # ===== 多模态相关参数 =====
    language_model_only: bool = MultiModalConfig.language_model_only  # 是否仅使用语言模型部分（忽略视觉等编码器）
    limit_mm_per_prompt: dict[str, int | dict[str, int]] = get_field(  # 每个提示中各种多模态输入的数量限制
        MultiModalConfig, "limit_per_prompt"
    )
    enable_mm_embeds: bool = MultiModalConfig.enable_mm_embeds  # 是否启用多模态嵌入
    interleave_mm_strings: bool = MultiModalConfig.interleave_mm_strings  # 是否交错处理多模态字符串
    media_io_kwargs: dict[str, dict[str, Any]] = get_field(  # 媒体 IO 处理的额外参数
        MultiModalConfig, "media_io_kwargs"
    )
    mm_processor_kwargs: dict[str, Any] | None = MultiModalConfig.mm_processor_kwargs  # 多模态处理器的额外参数
    mm_processor_cache_gb: float = MultiModalConfig.mm_processor_cache_gb  # 多模态处理器缓存大小（GB）
    mm_processor_cache_type: MMCacheType | None = (  # 多模态处理器缓存类型
        MultiModalConfig.mm_processor_cache_type
    )
    mm_shm_cache_max_object_size_mb: int = (  # 共享内存缓存单个对象最大大小（MB）
        MultiModalConfig.mm_shm_cache_max_object_size_mb
    )
    mm_encoder_only: bool = MultiModalConfig.mm_encoder_only  # 是否仅使用多模态编码器
    mm_encoder_tp_mode: MMEncoderTPMode = MultiModalConfig.mm_encoder_tp_mode  # 多模态编码器张量并行模式
    mm_encoder_attn_backend: AttentionBackendEnum | str | None = (  # 多模态编码器注意力后端
        MultiModalConfig.mm_encoder_attn_backend
    )
    io_processor_plugin: str | None = None  # IO 处理器插件名称
    skip_mm_profiling: bool = MultiModalConfig.skip_mm_profiling  # 是否跳过多模态性能分析
    video_pruning_rate: float | None = MultiModalConfig.video_pruning_rate  # 视频帧剪枝率
    # ===== LoRA 适配器相关参数 =====
    enable_lora: bool = False  # 是否启用 LoRA 适配器支持
    max_loras: int = LoRAConfig.max_loras  # 同时加载的最大 LoRA 适配器数量
    max_lora_rank: MaxLoRARanks = LoRAConfig.max_lora_rank  # LoRA 最大秩
    default_mm_loras: dict[str, str] | None = LoRAConfig.default_mm_loras  # 默认的模态特定 LoRA 适配器
    fully_sharded_loras: bool = LoRAConfig.fully_sharded_loras  # 是否完全分片 LoRA 权重
    max_cpu_loras: int | None = LoRAConfig.max_cpu_loras  # CPU 上缓存的最大 LoRA 数量
    lora_dtype: str | torch.dtype | None = LoRAConfig.lora_dtype  # LoRA 权重数据类型
    enable_tower_connector_lora: bool = LoRAConfig.enable_tower_connector_lora  # 是否启用 tower connector LoRA
    specialize_active_lora: bool = LoRAConfig.specialize_active_lora  # 是否特化活跃的 LoRA

    ray_workers_use_nsight: bool = ParallelConfig.ray_workers_use_nsight  # Ray 工作器是否使用 Nsight 分析
    num_gpu_blocks_override: int | None = CacheConfig.num_gpu_blocks_override  # 手动覆盖 GPU 缓存块数量
    model_loader_extra_config: dict = get_field(LoadConfig, "model_loader_extra_config")  # 模型加载器额外配置
    ignore_patterns: str | list[str] = get_field(LoadConfig, "ignore_patterns")  # 模型加载时忽略的文件模式

    # ===== 分块预填充相关参数 =====
    enable_chunked_prefill: bool | None = None  # 是否启用分块预填充，None 表示自动决定
    disable_chunked_mm_input: bool = SchedulerConfig.disable_chunked_mm_input  # 是否禁用多模态输入分块

    disable_hybrid_kv_cache_manager: bool | None = (  # 是否禁用混合 KV 缓存管理器
        SchedulerConfig.disable_hybrid_kv_cache_manager
    )

    # ===== 结构化输出相关参数 =====
    structured_outputs_config: StructuredOutputsConfig = get_field(  # 结构化输出配置
        VllmConfig, "structured_outputs_config"
    )
    reasoning_parser: str = StructuredOutputsConfig.reasoning_parser  # 推理解析器名称
    reasoning_parser_plugin: str | None = None  # 推理解析器插件

    # ===== 推测解码相关参数 =====
    speculative_config: dict[str, Any] | None = None  # 推测解码配置（JSON 字典或 None）

    # ===== 可观测性相关参数 =====
    show_hidden_metrics_for_version: str | None = (  # 显示隐藏指标的版本号
        ObservabilityConfig.show_hidden_metrics_for_version
    )
    otlp_traces_endpoint: str | None = ObservabilityConfig.otlp_traces_endpoint  # OpenTelemetry 追踪端点 URL
    collect_detailed_traces: list[DetailedTraceModules] | None = (  # 需要收集详细追踪的模块列表
        ObservabilityConfig.collect_detailed_traces
    )
    kv_cache_metrics: bool = ObservabilityConfig.kv_cache_metrics  # 是否收集 KV 缓存指标
    kv_cache_metrics_sample: float = get_field(  # KV 缓存指标采样率
        ObservabilityConfig, "kv_cache_metrics_sample"
    )
    cudagraph_metrics: bool = ObservabilityConfig.cudagraph_metrics  # 是否收集 CUDAGraph 指标
    enable_layerwise_nvtx_tracing: bool = (  # 是否启用逐层 NVTX 追踪
        ObservabilityConfig.enable_layerwise_nvtx_tracing
    )
    enable_mfu_metrics: bool = ObservabilityConfig.enable_mfu_metrics  # 是否启用 MFU（模型浮点运算利用率）指标
    enable_logging_iteration_details: bool = (  # 是否启用迭代详情日志
        ObservabilityConfig.enable_logging_iteration_details
    )
    enable_mm_processor_stats: bool = ObservabilityConfig.enable_mm_processor_stats  # 是否启用多模态处理器统计
    # ===== 调度策略参数 =====
    scheduling_policy: SchedulerPolicy = SchedulerConfig.policy  # 调度策略（fcfs/priority 等）
    scheduler_cls: str | type[object] | None = SchedulerConfig.scheduler_cls  # 自定义调度器类

    # ===== 池化与编译配置 =====
    pooler_config: PoolerConfig | None = ModelConfig.pooler_config  # 池化层配置（用于嵌入模型）
    compilation_config: CompilationConfig = get_field(VllmConfig, "compilation_config")  # 编译优化配置
    attention_config: AttentionConfig = get_field(VllmConfig, "attention_config")  # 注意力机制配置
    kernel_config: KernelConfig = get_field(VllmConfig, "kernel_config")  # 内核配置
    enable_flashinfer_autotune: bool = get_field(  # 是否启用 FlashInfer 自动调优
        KernelConfig, "enable_flashinfer_autotune"
    )
    worker_cls: str = ParallelConfig.worker_cls  # 自定义工作器类路径
    worker_extension_cls: str = ParallelConfig.worker_extension_cls  # 自定义工作器扩展类路径

    profiler_config: ProfilerConfig = get_field(VllmConfig, "profiler_config")  # 性能分析器配置

    # ===== KV 传输与事件配置 =====
    kv_transfer_config: KVTransferConfig | None = None  # KV 缓存传输配置（用于分离式推理）
    kv_events_config: KVEventsConfig | None = None  # KV 事件配置

    ec_transfer_config: ECTransferConfig | None = None  # EC 传输配置

    # ===== 生成配置参数 =====
    generation_config: str = ModelConfig.generation_config  # 生成配置来源（auto/model 等）
    enable_sleep_mode: bool = ModelConfig.enable_sleep_mode  # 是否启用睡眠模式（空闲时释放 GPU 内存）
    override_generation_config: dict[str, Any] = get_field(  # 覆盖生成配置的参数
        ModelConfig, "override_generation_config"
    )
    model_impl: str = ModelConfig.model_impl  # 模型实现方式（auto/vllm/transformers 等）
    override_attention_dtype: str | None = ModelConfig.override_attention_dtype  # 覆盖注意力计算数据类型
    attention_backend: AttentionBackendEnum | None = AttentionConfig.backend  # 注意力后端（flash_attn/flashinfer 等）

    # ===== Mamba 模型相关参数 =====
    calculate_kv_scales: bool = CacheConfig.calculate_kv_scales  # 是否计算 KV 缓存缩放因子
    mamba_cache_dtype: MambaDType = CacheConfig.mamba_cache_dtype  # Mamba 缓存数据类型
    mamba_ssm_cache_dtype: MambaDType = CacheConfig.mamba_ssm_cache_dtype  # Mamba SSM 缓存数据类型
    mamba_block_size: int | None = get_field(CacheConfig, "mamba_block_size")  # Mamba 缓存块大小
    mamba_cache_mode: MambaCacheMode = CacheConfig.mamba_cache_mode  # Mamba 缓存模式

    additional_config: dict[str, Any] = get_field(VllmConfig, "additional_config")  # 额外的自定义配置

    # ===== 模型加载参数 =====
    use_tqdm_on_load: bool = LoadConfig.use_tqdm_on_load  # 加载时是否显示进度条
    pt_load_map_location: str | dict[str, str] = LoadConfig.pt_load_map_location  # PyTorch 加载时的设备映射

    logits_processors: list[str | type[LogitsProcessor]] | None = (  # 自定义 logits 处理器类型列表
        ModelConfig.logits_processors
    )
    """自定义 logits 处理器类型"""

    async_scheduling: bool | None = SchedulerConfig.async_scheduling  # 是否启用异步调度

    stream_interval: int = SchedulerConfig.stream_interval  # 流式输出的间隔步数

    kv_sharing_fast_prefill: bool = CacheConfig.kv_sharing_fast_prefill  # 是否启用 KV 共享快速预填充
    optimization_level: OptimizationLevel = VllmConfig.optimization_level  # 优化级别
    performance_mode: PerformanceMode = VllmConfig.performance_mode  # 性能模式（throughput/latency 等）

    kv_offloading_size: float | None = CacheConfig.kv_offloading_size  # KV 缓存卸载大小
    kv_offloading_backend: KVOffloadingBackend = CacheConfig.kv_offloading_backend  # KV 缓存卸载后端
    tokens_only: bool = False  # 是否仅处理 token（跳过分词器初始化）

    shutdown_timeout: int = 0  # 关机超时时间（秒），0 表示立即终止

    weight_transfer_config: WeightTransferConfig | None = get_field(  # 权重传输配置
        VllmConfig,
        "weight_transfer_config",
    )

    fail_on_environ_validation: bool = False  # 环境验证失败时是否抛出错误

    def __post_init__(self):
        """数据类初始化后的处理方法。

        在数据类的 __init__ 执行完毕后自动调用，负责：
        1. 将字典类型的配置参数转换为对应的配置对象
        2. 加载通用插件
        3. 在 HuggingFace 离线模式下，将模型 ID 替换为本地路径
        """
        # 支持 EngineArgs(compilation_config={...}) 的写法
        # 无需手动构造 CompilationConfig 对象
        if isinstance(self.compilation_config, dict):  # 如果编译配置是字典
            self.compilation_config = CompilationConfig(**self.compilation_config)  # 转换为 CompilationConfig 对象
        if isinstance(self.attention_config, dict):  # 如果注意力配置是字典
            self.attention_config = AttentionConfig(**self.attention_config)  # 转换为 AttentionConfig 对象
        if isinstance(self.kernel_config, dict):  # 如果内核配置是字典
            self.kernel_config = KernelConfig(**self.kernel_config)  # 转换为 KernelConfig 对象
        if isinstance(self.eplb_config, dict):  # 如果 EPLB 配置是字典
            self.eplb_config = EPLBConfig(**self.eplb_config)  # 转换为 EPLBConfig 对象
        if isinstance(self.weight_transfer_config, dict):  # 如果权重传输配置是字典
            self.weight_transfer_config = WeightTransferConfig(  # 转换为 WeightTransferConfig 对象
                **self.weight_transfer_config
            )
        # 加载通用插件（插件可能会添加新的量化方法、设备类型等）
        from vllm.plugins import load_general_plugins

        load_general_plugins()  # 执行插件加载
        # 当使用 HuggingFace 离线模式时，将模型和分词器 ID 替换为本地模型路径
        if huggingface_hub.constants.HF_HUB_OFFLINE:  # 检查是否为离线模式
            model_id = self.model  # 保存原始模型 ID
            self.model = get_model_path(self.model, self.revision)  # 获取本地模型路径
            if model_id is not self.model:  # 如果路径发生了变化
                logger.info(  # 记录日志：模型 ID 已替换为本地路径
                    "HF_HUB_OFFLINE is True, replace model_id [%s] to model_path [%s]",
                    model_id,
                    self.model,
                )
            if self.tokenizer is not None:  # 如果指定了分词器
                tokenizer_id = self.tokenizer  # 保存原始分词器 ID
                self.tokenizer = get_model_path(self.tokenizer, self.tokenizer_revision)  # 获取本地分词器路径
                if tokenizer_id is not self.tokenizer:  # 如果路径发生了变化
                    logger.info(  # 记录日志：分词器 ID 已替换为本地路径
                        "HF_HUB_OFFLINE is True, replace tokenizer_id [%s] "
                        "to tokenizer_path [%s]",
                        tokenizer_id,
                        self.tokenizer,
                    )

    @staticmethod  # 静态方法装饰器，无需类实例即可调用
    def add_cli_args(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
        """添加 vLLM 引擎的共享命令行参数。

        该方法将所有引擎参数注册到 argparse 解析器中，
        按配置类别分组（模型、加载、并行、缓存、调度等）。

        参数:
            parser: 灵活的命令行参数解析器
        返回:
            添加了所有参数后的解析器
        """

        # ===== 模型参数组 =====
        model_kwargs = get_kwargs(ModelConfig)  # 获取 ModelConfig 的 argparse 参数
        model_group = parser.add_argument_group(  # 创建模型参数组
            title="ModelConfig",
            description=ModelConfig.__doc__,
        )
        if not ("serve" in sys.argv[1:] and "--help" in sys.argv[1:]):  # 在 serve --help 模式下隐藏 --model 参数
            model_group.add_argument("--model", **model_kwargs["model"])
        model_group.add_argument("--runner", **model_kwargs["runner"])
        model_group.add_argument("--convert", **model_kwargs["convert"])
        model_group.add_argument("--tokenizer", **model_kwargs["tokenizer"])
        model_group.add_argument("--tokenizer-mode", **model_kwargs["tokenizer_mode"])
        model_group.add_argument(
            "--trust-remote-code", **model_kwargs["trust_remote_code"]
        )
        model_group.add_argument("--dtype", **model_kwargs["dtype"])
        model_group.add_argument("--seed", **model_kwargs["seed"])
        model_group.add_argument("--hf-config-path", **model_kwargs["hf_config_path"])
        model_group.add_argument(
            "--allowed-local-media-path", **model_kwargs["allowed_local_media_path"]
        )
        model_group.add_argument(
            "--allowed-media-domains", **model_kwargs["allowed_media_domains"]
        )
        model_group.add_argument("--revision", **model_kwargs["revision"])
        model_group.add_argument("--code-revision", **model_kwargs["code_revision"])
        model_group.add_argument(
            "--tokenizer-revision", **model_kwargs["tokenizer_revision"]
        )
        model_group.add_argument("--max-model-len", **model_kwargs["max_model_len"])
        model_group.add_argument("--quantization", "-q", **model_kwargs["quantization"])
        model_group.add_argument(
            "--allow-deprecated-quantization",
            **model_kwargs["allow_deprecated_quantization"],
        )
        model_group.add_argument("--enforce-eager", **model_kwargs["enforce_eager"])
        model_group.add_argument(
            "--enable-return-routed-experts",
            **model_kwargs["enable_return_routed_experts"],
        )
        model_group.add_argument("--max-logprobs", **model_kwargs["max_logprobs"])
        model_group.add_argument("--logprobs-mode", **model_kwargs["logprobs_mode"])
        model_group.add_argument(
            "--disable-sliding-window", **model_kwargs["disable_sliding_window"]
        )
        model_group.add_argument(
            "--disable-cascade-attn", **model_kwargs["disable_cascade_attn"]
        )
        model_group.add_argument(
            "--skip-tokenizer-init", **model_kwargs["skip_tokenizer_init"]
        )
        model_group.add_argument(
            "--enable-prompt-embeds", **model_kwargs["enable_prompt_embeds"]
        )
        model_group.add_argument(
            "--served-model-name", **model_kwargs["served_model_name"]
        )
        model_group.add_argument("--config-format", **model_kwargs["config_format"])
        # This one is a special case because it can bool
        # or str. TODO: Handle this in get_kwargs
        model_group.add_argument(
            "--hf-token",
            type=str,
            nargs="?",
            const=True,
            default=model_kwargs["hf_token"]["default"],
            help=model_kwargs["hf_token"]["help"],
        )
        model_group.add_argument("--hf-overrides", **model_kwargs["hf_overrides"])
        model_group.add_argument("--pooler-config", **model_kwargs["pooler_config"])
        model_group.add_argument(
            "--generation-config", **model_kwargs["generation_config"]
        )
        model_group.add_argument(
            "--override-generation-config", **model_kwargs["override_generation_config"]
        )
        model_group.add_argument(
            "--enable-sleep-mode", **model_kwargs["enable_sleep_mode"]
        )
        model_group.add_argument("--model-impl", **model_kwargs["model_impl"])
        model_group.add_argument(
            "--override-attention-dtype", **model_kwargs["override_attention_dtype"]
        )
        model_group.add_argument(
            "--logits-processors", **model_kwargs["logits_processors"]
        )
        model_group.add_argument(
            "--io-processor-plugin", **model_kwargs["io_processor_plugin"]
        )

        # ===== 模型加载参数组 =====
        load_kwargs = get_kwargs(LoadConfig)  # 获取 LoadConfig 的 argparse 参数
        load_group = parser.add_argument_group(  # 创建模型加载参数组
            title="LoadConfig",
            description=LoadConfig.__doc__,
        )
        load_group.add_argument("--load-format", **load_kwargs["load_format"])
        load_group.add_argument("--download-dir", **load_kwargs["download_dir"])
        load_group.add_argument(
            "--safetensors-load-strategy", **load_kwargs["safetensors_load_strategy"]
        )
        load_group.add_argument(
            "--model-loader-extra-config", **load_kwargs["model_loader_extra_config"]
        )
        load_group.add_argument("--ignore-patterns", **load_kwargs["ignore_patterns"])
        load_group.add_argument("--use-tqdm-on-load", **load_kwargs["use_tqdm_on_load"])
        load_group.add_argument(
            "--pt-load-map-location", **load_kwargs["pt_load_map_location"]
        )

        # ===== 注意力参数组 =====
        attention_kwargs = get_kwargs(AttentionConfig)  # 获取 AttentionConfig 的 argparse 参数
        attention_group = parser.add_argument_group(  # 创建注意力参数组
            title="AttentionConfig",
            description=AttentionConfig.__doc__,
        )
        attention_group.add_argument(
            "--attention-backend", **attention_kwargs["backend"]
        )

        # ===== 结构化输出参数组 =====
        structured_outputs_kwargs = get_kwargs(StructuredOutputsConfig)  # 获取结构化输出配置的 argparse 参数
        structured_outputs_group = parser.add_argument_group(  # 创建结构化输出参数组
            title="StructuredOutputsConfig",
            description=StructuredOutputsConfig.__doc__,
        )
        structured_outputs_group.add_argument(
            "--reasoning-parser",
            # Choices need to be validated after parsing to include plugins
            **structured_outputs_kwargs["reasoning_parser"],
        )
        structured_outputs_group.add_argument(
            "--reasoning-parser-plugin",
            **structured_outputs_kwargs["reasoning_parser_plugin"],
        )

        # ===== 并行参数组 =====
        parallel_kwargs = get_kwargs(ParallelConfig)  # 获取 ParallelConfig 的 argparse 参数
        parallel_group = parser.add_argument_group(  # 创建并行参数组
            title="ParallelConfig",
            description=ParallelConfig.__doc__,
        )
        parallel_group.add_argument(
            "--distributed-executor-backend",
            **parallel_kwargs["distributed_executor_backend"],
        )
        parallel_group.add_argument(
            "--pipeline-parallel-size",
            "-pp",
            **parallel_kwargs["pipeline_parallel_size"],
        )
        parallel_group.add_argument("--master-addr", **parallel_kwargs["master_addr"])
        parallel_group.add_argument("--master-port", **parallel_kwargs["master_port"])
        parallel_group.add_argument("--nnodes", "-n", **parallel_kwargs["nnodes"])
        parallel_group.add_argument("--node-rank", "-r", **parallel_kwargs["node_rank"])
        parallel_group.add_argument(
            "--distributed-timeout-seconds",
            **parallel_kwargs["distributed_timeout_seconds"],
        )
        parallel_group.add_argument(
            "--tensor-parallel-size", "-tp", **parallel_kwargs["tensor_parallel_size"]
        )
        parallel_group.add_argument(
            "--decode-context-parallel-size",
            "-dcp",
            **parallel_kwargs["decode_context_parallel_size"],
        )
        parallel_group.add_argument(
            "--dcp-comm-backend",
            **parallel_kwargs["dcp_comm_backend"],
        )
        parallel_group.add_argument(
            "--dcp-kv-cache-interleave-size",
            **parallel_kwargs["dcp_kv_cache_interleave_size"],
        )
        parallel_group.add_argument(
            "--cp-kv-cache-interleave-size",
            **parallel_kwargs["cp_kv_cache_interleave_size"],
        )
        parallel_group.add_argument(
            "--prefill-context-parallel-size",
            "-pcp",
            **parallel_kwargs["prefill_context_parallel_size"],
        )
        parallel_group.add_argument(
            "--data-parallel-size", "-dp", **parallel_kwargs["data_parallel_size"]
        )
        parallel_group.add_argument(
            "--data-parallel-rank",
            "-dpn",
            type=int,
            help="Data parallel rank of this instance. "
            "When set, enables external load balancer mode.",
        )
        parallel_group.add_argument(
            "--data-parallel-start-rank",
            "-dpr",
            type=int,
            help="Starting data parallel rank for secondary nodes.",
        )
        parallel_group.add_argument(
            "--data-parallel-size-local",
            "-dpl",
            type=int,
            help="Number of data parallel replicas to run on this node.",
        )
        parallel_group.add_argument(
            "--data-parallel-address",
            "-dpa",
            type=str,
            help="Address of data parallel cluster head-node.",
        )
        parallel_group.add_argument(
            "--data-parallel-rpc-port",
            "-dpp",
            type=int,
            help="Port for data parallel RPC communication.",
        )
        parallel_group.add_argument(
            "--data-parallel-backend",
            "-dpb",
            type=str,
            default="mp",
            help='Backend for data parallel, either "mp" or "ray".',
        )
        parallel_group.add_argument(
            "--data-parallel-hybrid-lb",
            "-dph",
            **parallel_kwargs["data_parallel_hybrid_lb"],
        )
        parallel_group.add_argument(
            "--data-parallel-external-lb",
            "-dpe",
            **parallel_kwargs["data_parallel_external_lb"],
        )
        parallel_group.add_argument(
            "--enable-expert-parallel",
            "-ep",
            **parallel_kwargs["enable_expert_parallel"],
        )
        parallel_group.add_argument(
            "--all2all-backend", **parallel_kwargs["all2all_backend"]
        )
        parallel_group.add_argument("--enable-dbo", **parallel_kwargs["enable_dbo"])
        parallel_group.add_argument(
            "--ubatch-size",
            **parallel_kwargs["ubatch_size"],
        )
        parallel_group.add_argument(
            "--enable-elastic-ep", **parallel_kwargs["enable_elastic_ep"]
        )
        parallel_group.add_argument(
            "--dbo-decode-token-threshold",
            **parallel_kwargs["dbo_decode_token_threshold"],
        )
        parallel_group.add_argument(
            "--dbo-prefill-token-threshold",
            **parallel_kwargs["dbo_prefill_token_threshold"],
        )
        parallel_group.add_argument(
            "--disable-nccl-for-dp-synchronization",
            **parallel_kwargs["disable_nccl_for_dp_synchronization"],
        )
        parallel_group.add_argument("--enable-eplb", **parallel_kwargs["enable_eplb"])
        parallel_group.add_argument("--eplb-config", **parallel_kwargs["eplb_config"])
        parallel_group.add_argument(
            "--expert-placement-strategy",
            **parallel_kwargs["expert_placement_strategy"],
        )

        parallel_group.add_argument(
            "--max-parallel-loading-workers",
            **parallel_kwargs["max_parallel_loading_workers"],
        )
        parallel_group.add_argument(
            "--ray-workers-use-nsight", **parallel_kwargs["ray_workers_use_nsight"]
        )
        parallel_group.add_argument(
            "--disable-custom-all-reduce",
            **parallel_kwargs["disable_custom_all_reduce"],
        )
        parallel_group.add_argument("--worker-cls", **parallel_kwargs["worker_cls"])
        parallel_group.add_argument(
            "--worker-extension-cls", **parallel_kwargs["worker_extension_cls"]
        )

        # ===== KV 缓存参数组 =====
        cache_kwargs = get_kwargs(CacheConfig)  # 获取 CacheConfig 的 argparse 参数
        cache_group = parser.add_argument_group(  # 创建缓存参数组
            title="CacheConfig",
            description=CacheConfig.__doc__,
        )
        cache_group.add_argument("--block-size", **cache_kwargs["block_size"])
        cache_group.add_argument(
            "--gpu-memory-utilization", **cache_kwargs["gpu_memory_utilization"]
        )
        cache_group.add_argument(
            "--kv-cache-memory-bytes", **cache_kwargs["kv_cache_memory_bytes"]
        )
        cache_group.add_argument("--kv-cache-dtype", **cache_kwargs["cache_dtype"])
        cache_group.add_argument(
            "--num-gpu-blocks-override", **cache_kwargs["num_gpu_blocks_override"]
        )
        cache_group.add_argument(
            "--enable-prefix-caching",
            **{
                **cache_kwargs["enable_prefix_caching"],
                "default": None,
            },
        )
        cache_group.add_argument(
            "--prefix-caching-hash-algo", **cache_kwargs["prefix_caching_hash_algo"]
        )
        cache_group.add_argument(
            "--calculate-kv-scales", **cache_kwargs["calculate_kv_scales"]
        )
        cache_group.add_argument(
            "--kv-sharing-fast-prefill", **cache_kwargs["kv_sharing_fast_prefill"]
        )
        cache_group.add_argument(
            "--mamba-cache-dtype", **cache_kwargs["mamba_cache_dtype"]
        )
        cache_group.add_argument(
            "--mamba-ssm-cache-dtype", **cache_kwargs["mamba_ssm_cache_dtype"]
        )
        cache_group.add_argument(
            "--mamba-block-size", **cache_kwargs["mamba_block_size"]
        )
        cache_group.add_argument(
            "--mamba-cache-mode", **cache_kwargs["mamba_cache_mode"]
        )
        cache_group.add_argument(
            "--kv-offloading-size", **cache_kwargs["kv_offloading_size"]
        )
        cache_group.add_argument(
            "--kv-offloading-backend", **cache_kwargs["kv_offloading_backend"]
        )

        # ===== 模型权重卸载参数组 =====
        offload_kwargs = get_kwargs(OffloadConfig)  # 获取 OffloadConfig 的 argparse 参数
        uva_kwargs = get_kwargs(UVAOffloadConfig)  # 获取 UVA 卸载配置的 argparse 参数
        prefetch_kwargs = get_kwargs(PrefetchOffloadConfig)  # 获取预取卸载配置的 argparse 参数
        offload_group = parser.add_argument_group(  # 创建卸载参数组
            title="OffloadConfig",
            description=OffloadConfig.__doc__,
        )
        offload_group.add_argument(
            "--offload-backend", **offload_kwargs["offload_backend"]
        )
        offload_group.add_argument("--cpu-offload-gb", **uva_kwargs["cpu_offload_gb"])
        offload_group.add_argument(
            "--cpu-offload-params", **uva_kwargs["cpu_offload_params"]
        )
        offload_group.add_argument(
            "--offload-group-size",
            **prefetch_kwargs["offload_group_size"],
        )
        offload_group.add_argument(
            "--offload-num-in-group",
            **prefetch_kwargs["offload_num_in_group"],
        )
        offload_group.add_argument(
            "--offload-prefetch-step",
            **prefetch_kwargs["offload_prefetch_step"],
        )
        offload_group.add_argument(
            "--offload-params", **prefetch_kwargs["offload_params"]
        )

        # ===== 多模态参数组 =====
        multimodal_kwargs = get_kwargs(MultiModalConfig)  # 获取 MultiModalConfig 的 argparse 参数
        multimodal_group = parser.add_argument_group(  # 创建多模态参数组
            title="MultiModalConfig",
            description=MultiModalConfig.__doc__,
        )
        multimodal_group.add_argument(
            "--language-model-only", **multimodal_kwargs["language_model_only"]
        )
        multimodal_group.add_argument(
            "--limit-mm-per-prompt", **multimodal_kwargs["limit_per_prompt"]
        )
        multimodal_group.add_argument(
            "--enable-mm-embeds", **multimodal_kwargs["enable_mm_embeds"]
        )
        multimodal_group.add_argument(
            "--media-io-kwargs", **multimodal_kwargs["media_io_kwargs"]
        )
        multimodal_group.add_argument(
            "--mm-processor-kwargs", **multimodal_kwargs["mm_processor_kwargs"]
        )
        multimodal_group.add_argument(
            "--mm-processor-cache-gb", **multimodal_kwargs["mm_processor_cache_gb"]
        )
        multimodal_group.add_argument(
            "--mm-processor-cache-type", **multimodal_kwargs["mm_processor_cache_type"]
        )
        multimodal_group.add_argument(
            "--mm-shm-cache-max-object-size-mb",
            **multimodal_kwargs["mm_shm_cache_max_object_size_mb"],
        )
        multimodal_group.add_argument(
            "--mm-encoder-only", **multimodal_kwargs["mm_encoder_only"]
        )
        multimodal_group.add_argument(
            "--mm-encoder-tp-mode", **multimodal_kwargs["mm_encoder_tp_mode"]
        )
        multimodal_group.add_argument(
            "--mm-encoder-attn-backend",
            **multimodal_kwargs["mm_encoder_attn_backend"],
        )
        multimodal_group.add_argument(
            "--interleave-mm-strings", **multimodal_kwargs["interleave_mm_strings"]
        )
        multimodal_group.add_argument(
            "--skip-mm-profiling", **multimodal_kwargs["skip_mm_profiling"]
        )

        multimodal_group.add_argument(
            "--video-pruning-rate", **multimodal_kwargs["video_pruning_rate"]
        )

        # ===== LoRA 适配器参数组 =====
        lora_kwargs = get_kwargs(LoRAConfig)  # 获取 LoRAConfig 的 argparse 参数
        lora_group = parser.add_argument_group(  # 创建 LoRA 参数组
            title="LoRAConfig",
            description=LoRAConfig.__doc__,
        )
        lora_group.add_argument(
            "--enable-lora",
            action=argparse.BooleanOptionalAction,
            help="If True, enable handling of LoRA adapters.",
        )
        lora_group.add_argument("--max-loras", **lora_kwargs["max_loras"])
        lora_group.add_argument("--max-lora-rank", **lora_kwargs["max_lora_rank"])
        lora_group.add_argument(
            "--lora-dtype",
            **lora_kwargs["lora_dtype"],
        )
        lora_group.add_argument(
            "--enable-tower-connector-lora",
            **lora_kwargs["enable_tower_connector_lora"],
        )
        lora_group.add_argument("--max-cpu-loras", **lora_kwargs["max_cpu_loras"])
        lora_group.add_argument(
            "--fully-sharded-loras", **lora_kwargs["fully_sharded_loras"]
        )
        lora_group.add_argument("--default-mm-loras", **lora_kwargs["default_mm_loras"])
        lora_group.add_argument(
            "--specialize-active-lora", **lora_kwargs["specialize_active_lora"]
        )

        # ===== 可观测性参数组 =====
        observability_kwargs = get_kwargs(ObservabilityConfig)  # 获取可观测性配置的 argparse 参数
        observability_group = parser.add_argument_group(  # 创建可观测性参数组
            title="ObservabilityConfig",
            description=ObservabilityConfig.__doc__,
        )
        observability_group.add_argument(
            "--show-hidden-metrics-for-version",
            **observability_kwargs["show_hidden_metrics_for_version"],
        )
        observability_group.add_argument(
            "--otlp-traces-endpoint", **observability_kwargs["otlp_traces_endpoint"]
        )
        # TODO: 将来需要将此特殊处理泛化
        choices = observability_kwargs["collect_detailed_traces"]["choices"]  # 获取追踪模块的选项列表
        metavar = f"{{{','.join(choices)}}}"  # 生成选项的元变量显示文本
        observability_kwargs["collect_detailed_traces"]["metavar"] = metavar  # 设置元变量
        observability_kwargs["collect_detailed_traces"]["choices"] += [  # 添加所有两两组合的排列作为有效选项
            ",".join(p) for p in permutations(get_args(DetailedTraceModules), r=2)
        ]
        observability_group.add_argument(
            "--collect-detailed-traces",
            **observability_kwargs["collect_detailed_traces"],
        )
        observability_group.add_argument(
            "--kv-cache-metrics", **observability_kwargs["kv_cache_metrics"]
        )
        observability_group.add_argument(
            "--kv-cache-metrics-sample",
            **observability_kwargs["kv_cache_metrics_sample"],
        )
        observability_group.add_argument(
            "--cudagraph-metrics",
            **observability_kwargs["cudagraph_metrics"],
        )
        observability_group.add_argument(
            "--enable-layerwise-nvtx-tracing",
            **observability_kwargs["enable_layerwise_nvtx_tracing"],
        )
        observability_group.add_argument(
            "--enable-mfu-metrics",
            **observability_kwargs["enable_mfu_metrics"],
        )
        observability_group.add_argument(
            "--enable-logging-iteration-details",
            **observability_kwargs["enable_logging_iteration_details"],
        )

        # ===== 调度器参数组 =====
        scheduler_kwargs = get_kwargs(SchedulerConfig)  # 获取 SchedulerConfig 的 argparse 参数
        scheduler_group = parser.add_argument_group(  # 创建调度器参数组
            title="SchedulerConfig",
            description=SchedulerConfig.__doc__,
        )
        scheduler_group.add_argument(
            "--max-num-batched-tokens",
            **{
                **scheduler_kwargs["max_num_batched_tokens"],
                "default": None,
            },
        )
        scheduler_group.add_argument(
            "--max-num-seqs",
            **{
                **scheduler_kwargs["max_num_seqs"],
                "default": None,
            },
        )
        scheduler_group.add_argument(
            "--max-num-partial-prefills", **scheduler_kwargs["max_num_partial_prefills"]
        )
        scheduler_group.add_argument(
            "--max-long-partial-prefills",
            **scheduler_kwargs["max_long_partial_prefills"],
        )
        scheduler_group.add_argument(
            "--long-prefill-token-threshold",
            **scheduler_kwargs["long_prefill_token_threshold"],
        )
        # 多步调度已被移除；相应的参数不再支持。
        scheduler_group.add_argument(
            "--scheduling-policy", **scheduler_kwargs["policy"]
        )
        scheduler_group.add_argument(
            "--enable-chunked-prefill",
            **{
                **scheduler_kwargs["enable_chunked_prefill"],
                "default": None,
            },
        )
        scheduler_group.add_argument(
            "--disable-chunked-mm-input", **scheduler_kwargs["disable_chunked_mm_input"]
        )
        scheduler_group.add_argument(
            "--scheduler-cls", **scheduler_kwargs["scheduler_cls"]
        )
        scheduler_group.add_argument(
            "--disable-hybrid-kv-cache-manager",
            **scheduler_kwargs["disable_hybrid_kv_cache_manager"],
        )
        scheduler_group.add_argument(
            "--async-scheduling", **scheduler_kwargs["async_scheduling"]
        )
        scheduler_group.add_argument(
            "--stream-interval", **scheduler_kwargs["stream_interval"]
        )

        # ===== 编译优化参数组 =====
        compilation_kwargs = get_kwargs(CompilationConfig)  # 获取 CompilationConfig 的 argparse 参数
        compilation_group = parser.add_argument_group(  # 创建编译参数组
            title="CompilationConfig",
            description=CompilationConfig.__doc__,
        )
        compilation_group.add_argument(
            "--cudagraph-capture-sizes", **compilation_kwargs["cudagraph_capture_sizes"]
        )
        compilation_group.add_argument(
            "--max-cudagraph-capture-size",
            **compilation_kwargs["max_cudagraph_capture_size"],
        )

        # ===== 内核参数组 =====
        kernel_kwargs = get_kwargs(KernelConfig)  # 获取 KernelConfig 的 argparse 参数
        kernel_group = parser.add_argument_group(  # 创建内核参数组
            title="KernelConfig",
            description=KernelConfig.__doc__,
        )
        kernel_group.add_argument(
            "--enable-flashinfer-autotune",
            **kernel_kwargs["enable_flashinfer_autotune"],
        )
        moe_backend_kwargs = kernel_kwargs["moe_backend"]  # 获取 MoE 后端参数
        moe_backend_kwargs["type"] = lambda s: s.lower().replace("-", "_")  # 自定义类型转换：小写化并将连字符替换为下划线
        kernel_group.add_argument("--moe-backend", **moe_backend_kwargs)

        # ===== vLLM 总配置参数组 =====
        vllm_kwargs = get_kwargs(VllmConfig)  # 获取 VllmConfig 的 argparse 参数
        vllm_group = parser.add_argument_group(  # 创建 vLLM 配置参数组
            title="VllmConfig",
            description=VllmConfig.__doc__,
        )
        # 我们在 create_engine_config 中使用其他配置的字段来构造 SpeculativeConfig。
        # 因此这里将类型设置为 JSON 字符串，以延迟 SpeculativeConfig 自带的 Pydantic 验证。
        vllm_kwargs["speculative_config"]["type"] = optional_type(json.loads)  # 将推测配置类型设为可选 JSON 解析
        vllm_group.add_argument(
            "--speculative-config", **vllm_kwargs["speculative_config"]
        )
        vllm_group.add_argument(
            "--kv-transfer-config", **vllm_kwargs["kv_transfer_config"]
        )
        vllm_group.add_argument("--kv-events-config", **vllm_kwargs["kv_events_config"])
        vllm_group.add_argument(
            "--ec-transfer-config", **vllm_kwargs["ec_transfer_config"]
        )
        vllm_group.add_argument(
            "--compilation-config", "-cc", **vllm_kwargs["compilation_config"]
        )
        vllm_group.add_argument(
            "--attention-config", "-ac", **vllm_kwargs["attention_config"]
        )
        vllm_group.add_argument("--kernel-config", **vllm_kwargs["kernel_config"])
        vllm_group.add_argument(
            "--additional-config", **vllm_kwargs["additional_config"]
        )
        vllm_group.add_argument(
            "--structured-outputs-config", **vllm_kwargs["structured_outputs_config"]
        )
        vllm_group.add_argument("--profiler-config", **vllm_kwargs["profiler_config"])
        vllm_group.add_argument(
            "--optimization-level", **vllm_kwargs["optimization_level"]
        )
        vllm_group.add_argument("--performance-mode", **vllm_kwargs["performance_mode"])
        vllm_group.add_argument(
            "--weight-transfer-config", **vllm_kwargs["weight_transfer_config"]
        )

        # ===== 其他参数 =====
        parser.add_argument(  # 添加禁用统计日志参数
            "--disable-log-stats",
            action="store_true",
            help="Disable logging statistics.",
        )

        parser.add_argument(
            "--aggregate-engine-logging",
            action="store_true",
            help="Log aggregate rather than per-engine statistics "
            "when using data parallelism.",
        )

        parser.add_argument(
            "--fail-on-environ-validation",
            help="If set, the engine will raise an error if "
            "environment validation fails.",
            default=False,
            action=argparse.BooleanOptionalAction,
        )

        parser.add_argument(
            "--shutdown-timeout",
            type=int,
            default=0,
            help="Shutdown timeout in seconds. 0 = abort, >0 = wait.",
        )

        return parser

    @classmethod  # 类方法装饰器，第一个参数为类本身
    def from_cli_args(cls, args: argparse.Namespace):
        """从命令行解析结果创建 EngineArgs 实例。

        参数:
            args: argparse 解析后的命名空间对象
        返回:
            根据命令行参数构造的 EngineArgs 实例
        """
        # 获取该数据类所有字段的名称列表
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # 从解析后的参数中提取对应属性值来构造引擎参数
        engine_args = cls(
            **{attr: getattr(args, attr) for attr in attrs if hasattr(args, attr)}
        )
        return engine_args  # 返回构造好的引擎参数实例

    def create_model_config(self) -> ModelConfig:
        """根据引擎参数创建模型配置对象。

        处理 GGUF 格式的特殊加载需求，并在非多进程模式下发出随机种子警告。

        返回:
            ModelConfig 模型配置实例
        """
        # GGUF 文件需要特定的模型加载器
        if is_gguf(self.model):  # 检查模型是否为 GGUF 格式
            self.quantization = self.load_format = "gguf"  # 设置量化方法和加载格式为 gguf

        if not envs.VLLM_ENABLE_V1_MULTIPROCESSING:  # 如果未启用 V1 多进程
            logger.warning(
                "The global random seed is set to %d. Since "
                "VLLM_ENABLE_V1_MULTIPROCESSING is set to False, this may "
                "affect the random state of the Python process that "
                "launched vLLM.",
                self.seed,
            )

        # 使用引擎参数构造 ModelConfig 对象
        return ModelConfig(
            model=self.model,  # 模型名称或路径
            model_weights=self.model_weights,  # 模型权重路径
            hf_config_path=self.hf_config_path,  # HuggingFace 配置路径
            runner=self.runner,  # 运行器类型
            convert=self.convert,  # 转换选项
            tokenizer=self.tokenizer,  # type: ignore[arg-type]  # 分词器
            tokenizer_mode=self.tokenizer_mode,  # 分词器模式
            trust_remote_code=self.trust_remote_code,  # 是否信任远程代码
            allowed_local_media_path=self.allowed_local_media_path,  # 允许的本地媒体路径
            allowed_media_domains=self.allowed_media_domains,  # 允许的媒体域名
            dtype=self.dtype,  # 数据类型
            seed=self.seed,  # 随机种子
            revision=self.revision,  # 模型版本
            code_revision=self.code_revision,  # 代码版本
            hf_token=self.hf_token,  # HuggingFace 令牌
            hf_overrides=self.hf_overrides,  # HuggingFace 配置覆盖
            tokenizer_revision=self.tokenizer_revision,  # 分词器版本
            max_model_len=self.max_model_len,  # 最大模型长度
            quantization=self.quantization,  # 量化方法
            allow_deprecated_quantization=self.allow_deprecated_quantization,  # 允许已弃用量化
            enforce_eager=self.enforce_eager,  # 强制 eager 模式
            enable_return_routed_experts=self.enable_return_routed_experts,  # 返回路由专家信息
            max_logprobs=self.max_logprobs,  # 最大对数概率数
            logprobs_mode=self.logprobs_mode,  # 对数概率模式
            disable_sliding_window=self.disable_sliding_window,  # 禁用滑动窗口
            disable_cascade_attn=self.disable_cascade_attn,  # 禁用级联注意力
            skip_tokenizer_init=self.skip_tokenizer_init,  # 跳过分词器初始化
            enable_prompt_embeds=self.enable_prompt_embeds,  # 启用提示嵌入
            served_model_name=self.served_model_name,  # 服务的模型名称
            language_model_only=self.language_model_only,  # 仅语言模型
            limit_mm_per_prompt=self.limit_mm_per_prompt,  # 每提示多模态限制
            enable_mm_embeds=self.enable_mm_embeds,  # 启用多模态嵌入
            interleave_mm_strings=self.interleave_mm_strings,  # 交错多模态字符串
            media_io_kwargs=self.media_io_kwargs,  # 媒体 IO 参数
            skip_mm_profiling=self.skip_mm_profiling,  # 跳过多模态分析
            config_format=self.config_format,  # 配置格式
            mm_processor_kwargs=self.mm_processor_kwargs,  # 多模态处理器参数
            mm_processor_cache_gb=self.mm_processor_cache_gb,  # 多模态缓存大小
            mm_processor_cache_type=self.mm_processor_cache_type,  # 多模态缓存类型
            mm_shm_cache_max_object_size_mb=self.mm_shm_cache_max_object_size_mb,  # 共享内存缓存对象最大大小
            mm_encoder_only=self.mm_encoder_only,  # 仅编码器
            mm_encoder_tp_mode=self.mm_encoder_tp_mode,  # 编码器张量并行模式
            mm_encoder_attn_backend=self.mm_encoder_attn_backend,  # 编码器注意力后端
            pooler_config=self.pooler_config,  # 池化配置
            generation_config=self.generation_config,  # 生成配置
            override_generation_config=self.override_generation_config,  # 覆盖生成配置
            enable_sleep_mode=self.enable_sleep_mode,  # 睡眠模式
            model_impl=self.model_impl,  # 模型实现
            override_attention_dtype=self.override_attention_dtype,  # 覆盖注意力数据类型
            logits_processors=self.logits_processors,  # logits 处理器
            video_pruning_rate=self.video_pruning_rate,  # 视频剪枝率
            io_processor_plugin=self.io_processor_plugin,  # IO 处理器插件
        )

    def validate_tensorizer_args(self):
        """验证并处理 tensorizer 相关的参数。

        将 model_loader_extra_config 中属于 TensorizerConfig 的字段
        移动到 tensorizer_config 子字典中。
        """
        from vllm.model_executor.model_loader.tensorizer import TensorizerConfig  # 导入 Tensorizer 配置类

        for key in self.model_loader_extra_config:  # 遍历额外配置中的所有键
            if key in TensorizerConfig._fields:  # 如果键属于 TensorizerConfig 的字段
                self.model_loader_extra_config["tensorizer_config"][key] = (  # 将其移动到 tensorizer_config 子字典
                    self.model_loader_extra_config[key]
                )

    def create_load_config(self) -> LoadConfig:
        """根据引擎参数创建模型加载配置对象。

        处理 bitsandbytes 和 tensorizer 加载格式的特殊需求。

        返回:
            LoadConfig 模型加载配置实例
        """
        if self.quantization == "bitsandbytes":  # 如果使用 bitsandbytes 量化
            self.load_format = "bitsandbytes"  # 需要使用对应的加载格式

        if self.load_format == "tensorizer":  # 如果使用 tensorizer 格式加载
            if hasattr(self.model_loader_extra_config, "to_serializable"):  # 如果配置有序列化方法
                self.model_loader_extra_config = (  # 转换为可序列化格式
                    self.model_loader_extra_config.to_serializable()
                )
            self.model_loader_extra_config["tensorizer_config"] = {}  # 初始化 tensorizer 配置字典
            self.model_loader_extra_config["tensorizer_config"]["tensorizer_dir"] = (  # 设置 tensorizer 目录为模型路径
                self.model
            )
            self.validate_tensorizer_args()  # 验证 tensorizer 参数

        # 构造并返回 LoadConfig 对象
        return LoadConfig(
            load_format=self.load_format,  # 加载格式
            download_dir=self.download_dir,  # 下载目录
            safetensors_load_strategy=self.safetensors_load_strategy,  # safetensors 加载策略
            model_loader_extra_config=self.model_loader_extra_config,  # 加载器额外配置
            ignore_patterns=self.ignore_patterns,  # 忽略的文件模式
            use_tqdm_on_load=self.use_tqdm_on_load,  # 加载时是否显示进度条
            pt_load_map_location=self.pt_load_map_location,  # PyTorch 加载设备映射
        )

    def create_speculative_config(
        self,
        target_model_config: ModelConfig,
        target_parallel_config: ParallelConfig,
    ) -> SpeculativeConfig | None:
        """根据 speculative_config 初始化并返回推测解码配置对象。

        该函数使用 speculative_config 创建 SpeculativeConfig 对象。
        speculative_config 可以通过 CLI 参数以 JSON 字符串形式提供，
        也可以从引擎直接以字典形式传入。

        参数:
            target_model_config: 目标模型配置
            target_parallel_config: 目标并行配置
        返回:
            SpeculativeConfig 实例，如果未配置推测解码则返回 None
        """
        if self.speculative_config is None:  # 如果未配置推测解码
            return None  # 返回 None

        # 注意(Shangming): 这些参数不是从 CLI 参数 '--speculative-config' 获取的，
        # 必须在创建引擎配置时传入。
        self.speculative_config.update(  # 更新推测配置，添加目标模型和并行配置
            {
                "target_model_config": target_model_config,  # 目标模型配置
                "target_parallel_config": target_parallel_config,  # 目标并行配置
            }
        )
        return SpeculativeConfig(**self.speculative_config)  # 构造并返回推测解码配置

    def create_engine_config(
        self,
        usage_context: UsageContext | None = None,
        headless: bool = False,
    ) -> VllmConfig:
        """创建 VllmConfig 总配置对象。

        这是引擎配置创建的核心方法，负责：
        1. 设备检测和环境验证
        2. 推测器模型检测和覆盖
        3. 创建模型、缓存、并行、调度等子配置
        4. 参数兼容性验证

        参数:
            usage_context: 使用场景上下文（LLM_CLASS/OPENAI_API_SERVER 等）
            headless: 是否为无头模式（无 API 服务器）
        返回:
            VllmConfig 总配置实例
        注意: 如果 VllmConfig 不兼容，会抛出错误。
        """
        current_platform.pre_register_and_update()  # 预注册并更新当前平台信息

        device_config = DeviceConfig(device=cast(Device, current_platform.device_type))  # 创建设备配置

        envs.validate_environ(self.fail_on_environ_validation)  # 验证环境变量

        # 在创建 ModelConfig 之前检查模型是否为推测器，并覆盖模型/分词器/配置
        # 这样配置会使用目标模型来创建
        # 跳过云存储模型（如 S3、GCS）的推测器检测，因为 HuggingFace 无法直接从 S3 URL 加载配置
        # S3 模型仍然可以通过显式 --speculative-config 使用推测器
        if not is_cloud_storage(self.model):  # 如果模型不在云存储中
            (self.model, self.tokenizer, self.speculative_config) = (  # 检测并处理推测器覆盖
                maybe_override_with_speculators(
                    model=self.model,  # 模型路径
                    tokenizer=self.tokenizer,  # 分词器路径
                    revision=self.revision,  # 版本
                    trust_remote_code=self.trust_remote_code,  # 是否信任远程代码
                    vllm_speculative_config=self.speculative_config,  # 推测配置
                )
            )

        model_config = self.create_model_config()  # 创建模型配置
        self.model = model_config.model  # 更新模型路径（可能被 ModelConfig 修改）
        self.model_weights = model_config.model_weights  # 更新模型权重路径
        self.tokenizer = model_config.tokenizer  # 更新分词器路径

        self._check_feature_supported()  # 检查特性支持情况
        self._set_default_chunked_prefill_and_prefix_caching_args(model_config)  # 设置分块预填充和前缀缓存的默认值
        self._set_default_max_num_seqs_and_batched_tokens_args(  # 设置最大序列数和批处理 token 数的默认值
            usage_context, model_config
        )

        sliding_window: int | None = None  # 初始化滑动窗口大小为 None
        if not is_interleaved(model_config.hf_text_config):  # 如果模型不是交错滑动窗口架构
            # 仅在模型全部使用滑动窗口时设置 CacheConfig.sliding_window
            # 否则 CacheConfig.sliding_window 会覆盖交错滑动窗口模型中的全局层
            sliding_window = model_config.get_sliding_window()  # 获取滑动窗口大小

        # 将 "auto" KV 缓存数据类型解析为模型配置中的实际值
        resolved_cache_dtype = resolve_kv_cache_dtype_string(
            self.kv_cache_dtype, model_config  # 解析 KV 缓存数据类型字符串
        )

        assert self.enable_prefix_caching is not None, (
            "enable_prefix_caching must be set by this point"
        )

        # 创建 KV 缓存配置
        cache_config = CacheConfig(
            block_size=self.block_size,  # type: ignore[arg-type]  # 缓存块大小
            gpu_memory_utilization=self.gpu_memory_utilization,  # GPU 内存利用率
            kv_cache_memory_bytes=self.kv_cache_memory_bytes,  # KV 缓存内存字节数
            cache_dtype=resolved_cache_dtype,  # type: ignore[arg-type]  # 已解析的缓存数据类型
            is_attention_free=model_config.is_attention_free,  # 模型是否无注意力（如 Mamba）
            num_gpu_blocks_override=self.num_gpu_blocks_override,  # GPU 块数覆盖
            sliding_window=sliding_window,  # 滑动窗口大小
            enable_prefix_caching=self.enable_prefix_caching,  # 是否启用前缀缓存
            prefix_caching_hash_algo=self.prefix_caching_hash_algo,  # 前缀缓存哈希算法
            calculate_kv_scales=self.calculate_kv_scales,  # 是否计算 KV 缩放因子
            kv_sharing_fast_prefill=self.kv_sharing_fast_prefill,  # KV 共享快速预填充
            mamba_cache_dtype=self.mamba_cache_dtype,  # Mamba 缓存数据类型
            mamba_ssm_cache_dtype=self.mamba_ssm_cache_dtype,  # Mamba SSM 缓存数据类型
            mamba_block_size=self.mamba_block_size,  # Mamba 块大小
            mamba_cache_mode=self.mamba_cache_mode,  # Mamba 缓存模式
            kv_offloading_size=self.kv_offloading_size,  # KV 卸载大小
            kv_offloading_backend=self.kv_offloading_backend,  # KV 卸载后端
        )

        # 获取 Ray 运行时环境（如果已初始化）
        ray_runtime_env = None  # 初始化 Ray 运行时环境为 None
        if is_ray_initialized():  # 如果 Ray 已初始化
            # Ray Serve LLM 在 Ray 任务上下文中调用 create_engine_config，
            # 因此我们检查 is_ray_initialized() 而不是 is_in_ray_actor()
            import ray  # 导入 Ray

            ray_runtime_env = ray.get_runtime_context().runtime_env  # 获取 Ray 运行时环境
            # 避免记录敏感的环境变量
            sanitized_env = ray_runtime_env.to_dict() if ray_runtime_env else {}  # 转换为字典
            if "env_vars" in sanitized_env:  # 如果包含环境变量
                sanitized_env["env_vars"] = {  # 将环境变量值替换为 ***
                    k: "***" for k in sanitized_env["env_vars"]
                }
            logger.info("Using ray runtime env (env vars redacted): %s", sanitized_env)  # 记录脱敏后的环境信息

        # 如果 Ray 已初始化且当前在 Ray actor 中，获取当前的放置组
        # 放置组会传递给生成的子进程
        placement_group = None  # 初始化放置组为 None
        if is_in_ray_actor():  # 如果当前在 Ray actor 中
            import ray  # 导入 Ray

            # 此调用会在 Ray 未初始化时自动初始化，但我们不应在此处这样做
            placement_group = ray.util.get_current_placement_group()  # 获取当前放置组

        # ===== 数据并行配置验证与推断 =====
        assert not headless or not self.data_parallel_hybrid_lb, (  # 无头模式下不支持混合负载均衡
            "data_parallel_hybrid_lb is not applicable in headless mode"
        )
        assert not (self.data_parallel_hybrid_lb and self.data_parallel_external_lb), (  # 两种负载均衡不能同时启用
            "data_parallel_hybrid_lb and data_parallel_external_lb cannot both be True."
        )
        assert self.data_parallel_backend == "mp" or self.nnodes == 1, (  # 多节点仅支持 mp 后端
            "nnodes > 1 is only supported with data_parallel_backend=mp"
        )
        inferred_data_parallel_rank = 0  # 初始化推断的数据并行排名为 0
        if self.nnodes > 1:  # 如果是多节点部署
            world_size = (  # 计算总的 world_size（所有并行维度的乘积）
                self.data_parallel_size
                * self.pipeline_parallel_size
                * self.tensor_parallel_size
            )
            world_size_within_dp = (  # 单个数据并行副本内的 world_size
                self.pipeline_parallel_size * self.tensor_parallel_size
            )
            local_world_size = world_size // self.nnodes  # 每个节点的 world_size
            assert world_size % self.nnodes == 0, (
                f"world_size={world_size} must be divisible by nnodes={self.nnodes}."
            )
            assert self.node_rank < self.nnodes, (
                f"node_rank={self.node_rank} must be less than nnodes={self.nnodes}."
            )
            inferred_data_parallel_rank = (  # 根据节点排名推断数据并行排名
                self.node_rank * local_world_size
            ) // world_size_within_dp
            if self.data_parallel_size > 1 and self.data_parallel_external_lb:  # 如果启用了外部负载均衡
                self.data_parallel_rank = inferred_data_parallel_rank
                logger.info(
                    "Inferred data_parallel_rank %d from node_rank %d for external lb",
                    self.data_parallel_rank,
                    self.node_rank,
                )
            elif self.data_parallel_size_local is None:  # 如果未设置本地数据并行大小
                # 为内部数据并行负载均衡推断本地数据并行大小
                self.data_parallel_size_local = max(
                    local_world_size // world_size_within_dp, 1  # 至少为 1
                )
        # 判断是否使用外部负载均衡（显式设置或指定了数据并行排名）
        data_parallel_external_lb = (
            self.data_parallel_external_lb or self.data_parallel_rank is not None
        )
        # 本地 DP 排名 = 1 时，使用纯外部负载均衡
        if data_parallel_external_lb:  # 如果启用外部负载均衡
            assert self.data_parallel_rank is not None, (
                "data_parallel_rank or node_rank must be specified if "
                "data_parallel_external_lb is enable."
            )
            assert self.data_parallel_size_local in (1, None), (
                "data_parallel_size_local must be 1 or None when data_parallel_rank "
                "is set"
            )
            data_parallel_size_local = 1  # 本地数据并行大小设为 1
            # 如果本地大小为 1，使用完全外部负载均衡
            self.data_parallel_hybrid_lb = False  # 禁用混合负载均衡
        elif self.data_parallel_size_local is not None:  # 如果指定了本地数据并行大小
            data_parallel_size_local = self.data_parallel_size_local  # 使用指定的本地数据并行大小

            if self.data_parallel_start_rank and not headless:  # 如果指定了起始排名且非无头模式
                # 推断为混合负载均衡模式
                self.data_parallel_hybrid_lb = True  # 启用混合负载均衡

            if self.data_parallel_hybrid_lb and data_parallel_size_local == 1:  # 如果混合 LB 但本地大小为 1
                # 本地大小为 1 时使用完全外部负载均衡
                logger.warning(
                    "data_parallel_hybrid_lb is not eligible when "
                    "data_parallel_size_local = 1, autoswitch to "
                    "data_parallel_external_lb."
                )
                data_parallel_external_lb = True  # 切换到外部负载均衡
                self.data_parallel_hybrid_lb = False  # 禁用混合负载均衡

            if data_parallel_size_local == self.data_parallel_size:  # 如果本地大小等于全局大小（单节点）
                # 单节点时禁用混合负载均衡模式
                self.data_parallel_hybrid_lb = False

            self.data_parallel_rank = (  # 设置数据并行排名
                self.data_parallel_start_rank or inferred_data_parallel_rank  # 使用指定的起始排名或推断的排名
            )
            if self.nnodes > 1:
                logger.info(
                    "Inferred data_parallel_rank %d from node_rank %d",
                    self.data_parallel_rank,
                    self.node_rank,
                )
        else:
            assert not self.data_parallel_hybrid_lb, (
                "data_parallel_size_local must be set to use data_parallel_hybrid_lb."
            )

            if self.data_parallel_backend == "ray" and (  # 如果是 Ray 后端且使用 span 放置策略
                envs.VLLM_RAY_DP_PACK_STRATEGY == "span"
            ):
                # DP 排名跨越多节点时，默认本地数据并行大小为 1
                data_parallel_size_local = 1
            else:
                # 否则本地 DP 大小默认为全局 DP 大小
                data_parallel_size_local = self.data_parallel_size

        # DP 地址，用于多节点场景的 torch distributed 组和 ZMQ 套接字
        if self.data_parallel_address is None:  # 如果未指定数据并行地址
            if self.data_parallel_backend == "ray":  # Ray 后端
                host_ip = get_ip()  # 获取本机 IP 地址
                logger.info(  # 记录日志：使用主机 IP 作为 Ray 数据并行地址
                    "Using host IP %s as ray-based data parallel address", host_ip
                )
                data_parallel_address = host_ip  # 使用主机 IP
            else:  # mp 后端
                assert self.data_parallel_backend == "mp", (
                    "data_parallel_backend can only be ray or mp, got %s",
                    self.data_parallel_backend,
                )
                data_parallel_address = (  # 使用主节点地址或默认数据并行主 IP
                    self.master_addr or ParallelConfig.data_parallel_master_ip
                )
        else:
            data_parallel_address = self.data_parallel_address  # 使用用户指定的地址

        # 此端口仅在有远程数据并行引擎时使用，
        # 否则使用本地 IPC 传输。
        data_parallel_rpc_port = (  # 数据并行 RPC 端口
            self.data_parallel_rpc_port  # 使用用户指定的端口
            if (self.data_parallel_rpc_port is not None)
            else ParallelConfig.data_parallel_rpc_port  # 否则使用默认端口
        )

        if self.tokens_only and not model_config.skip_tokenizer_init:  # 如果是仅 token 模式但未跳过分词器初始化
            model_config.skip_tokenizer_init = True
            logger.info("Skipping tokenizer initialization for tokens-only mode.")

        # 创建并行配置
        parallel_config = ParallelConfig(
            pipeline_parallel_size=self.pipeline_parallel_size,  # 流水线并行大小
            tensor_parallel_size=self.tensor_parallel_size,  # 张量并行大小
            prefill_context_parallel_size=self.prefill_context_parallel_size,  # 预填充上下文并行大小
            data_parallel_size=self.data_parallel_size,  # 数据并行大小
            data_parallel_rank=self.data_parallel_rank or 0,  # 数据并行排名（默认 0）
            data_parallel_external_lb=data_parallel_external_lb,  # 是否使用外部负载均衡
            data_parallel_size_local=data_parallel_size_local,  # 本地数据并行大小
            master_addr=self.master_addr,  # 主节点地址
            master_port=self.master_port,  # 主节点端口
            nnodes=self.nnodes,  # 节点数量
            node_rank=self.node_rank,  # 节点排名
            distributed_timeout_seconds=self.distributed_timeout_seconds,  # 分布式超时
            data_parallel_master_ip=data_parallel_address,  # 数据并行主 IP
            data_parallel_rpc_port=data_parallel_rpc_port,  # 数据并行 RPC 端口
            data_parallel_backend=self.data_parallel_backend,  # 数据并行后端
            data_parallel_hybrid_lb=self.data_parallel_hybrid_lb,  # 混合负载均衡
            is_moe_model=model_config.is_moe,  # 是否为 MoE 模型
            enable_expert_parallel=self.enable_expert_parallel,  # 专家并行
            all2all_backend=self.all2all_backend,  # All-to-All 后端
            enable_elastic_ep=self.enable_elastic_ep,  # 弹性专家并行
            enable_dbo=self.enable_dbo,  # 分离式批处理编排
            ubatch_size=self.ubatch_size,  # 微批次大小
            dbo_decode_token_threshold=self.dbo_decode_token_threshold,  # DBO 解码阈值
            dbo_prefill_token_threshold=self.dbo_prefill_token_threshold,  # DBO 预填充阈值
            disable_nccl_for_dp_synchronization=self.disable_nccl_for_dp_synchronization,  # 禁用 NCCL DP 同步
            enable_eplb=self.enable_eplb,  # EPLB 负载均衡
            eplb_config=self.eplb_config,  # EPLB 配置
            expert_placement_strategy=self.expert_placement_strategy,  # 专家放置策略
            max_parallel_loading_workers=self.max_parallel_loading_workers,  # 最大并行加载工作器
            disable_custom_all_reduce=self.disable_custom_all_reduce,  # 禁用自定义 AllReduce
            ray_workers_use_nsight=self.ray_workers_use_nsight,  # Ray 使用 Nsight
            ray_runtime_env=ray_runtime_env,  # Ray 运行时环境
            placement_group=placement_group,  # Ray 放置组
            distributed_executor_backend=self.distributed_executor_backend,  # 分布式执行器后端
            worker_cls=self.worker_cls,  # 工作器类
            worker_extension_cls=self.worker_extension_cls,  # 工作器扩展类
            decode_context_parallel_size=self.decode_context_parallel_size,  # 解码上下文并行大小
            dcp_comm_backend=self.dcp_comm_backend,  # DCP 通信后端
            dcp_kv_cache_interleave_size=self.dcp_kv_cache_interleave_size,  # DCP KV 缓存交错大小
            cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,  # CP KV 缓存交错大小
            _api_process_count=self._api_process_count,  # API 进程总数
            _api_process_rank=self._api_process_rank,  # API 进程排名
        )

        # 创建推测解码配置
        speculative_config = self.create_speculative_config(
            target_model_config=model_config,  # 目标模型配置
            target_parallel_config=parallel_config,  # 目标并行配置
        )

        # 验证必要参数已设置
        assert self.max_num_batched_tokens is not None, (  # 确保最大批处理 token 数已设置
            "max_num_batched_tokens must be set by this point"
        )
        assert self.max_num_seqs is not None, "max_num_seqs must be set by this point"
        assert self.enable_chunked_prefill is not None, (
            "enable_chunked_prefill must be set by this point"
        )
        assert model_config.max_model_len is not None, (
            "max_model_len must be set by this point"
        )
        # 创建调度器配置
        scheduler_config = SchedulerConfig(
            runner_type=model_config.runner_type,  # 运行器类型
            max_num_batched_tokens=self.max_num_batched_tokens,  # 最大批处理 token 数
            max_num_seqs=self.max_num_seqs,  # 最大序列数
            max_model_len=model_config.max_model_len,  # 最大模型长度
            enable_chunked_prefill=self.enable_chunked_prefill,  # 分块预填充
            disable_chunked_mm_input=self.disable_chunked_mm_input,  # 禁用多模态输入分块
            is_multimodal_model=model_config.is_multimodal_model,  # 是否为多模态模型
            is_encoder_decoder=model_config.is_encoder_decoder,  # 是否为编码器-解码器模型
            policy=self.scheduling_policy,  # 调度策略
            scheduler_cls=self.scheduler_cls,  # 自定义调度器类
            max_num_partial_prefills=self.max_num_partial_prefills,  # 最大部分预填充数
            max_long_partial_prefills=self.max_long_partial_prefills,  # 最大长预填充数
            long_prefill_token_threshold=self.long_prefill_token_threshold,  # 长预填充阈值
            disable_hybrid_kv_cache_manager=self.disable_hybrid_kv_cache_manager,  # 禁用混合 KV 缓存管理器
            async_scheduling=self.async_scheduling,  # 异步调度
            stream_interval=self.stream_interval,  # 流式间隔
        )

        if not model_config.is_multimodal_model and self.default_mm_loras:
            raise ValueError(
                "Default modality-specific LoRA(s) were provided for a "
                "non multimodal model"
            )

        # 创建 LoRA 配置（仅在启用 LoRA 时创建，否则为 None）
        lora_config = (
            LoRAConfig(
                max_lora_rank=self.max_lora_rank,  # 最大 LoRA 秩
                max_loras=self.max_loras,  # 最大 LoRA 数量
                default_mm_loras=self.default_mm_loras,  # 默认多模态 LoRA
                fully_sharded_loras=self.fully_sharded_loras,  # 完全分片 LoRA
                lora_dtype=self.lora_dtype,  # LoRA 数据类型
                enable_tower_connector_lora=self.enable_tower_connector_lora,  # tower connector LoRA
                specialize_active_lora=self.specialize_active_lora,  # 特化活跃 LoRA
                max_cpu_loras=self.max_cpu_loras  # CPU 上最大 LoRA 数（大于 0 时才设置）
                if self.max_cpu_loras and self.max_cpu_loras > 0
                else None,
            )
            if self.enable_lora
            else None
        )

        if (
            lora_config is not None
            and speculative_config is not None
            and scheduler_config.max_num_batched_tokens
            < (
                scheduler_config.max_num_seqs
                * (speculative_config.num_speculative_tokens + 1)
            )
        ):
            raise ValueError(
                "Consider increasing max_num_batched_tokens or "
                "decreasing num_speculative_tokens"
            )

        # bitsandbytes 预量化模型需要特定的模型加载器
        if model_config.quantization == "bitsandbytes":  # 如果模型量化方式是 bitsandbytes
            self.quantization = self.load_format = "bitsandbytes"  # 设置量化和加载格式

        # 注意力配置覆盖处理
        attention_config = copy.deepcopy(self.attention_config)  # 深拷贝注意力配置以避免修改原始对象
        if self.attention_backend is not None:  # 如果指定了注意力后端
            if attention_config.backend is not None:  # 如果配置中也指定了后端（冲突）
                raise ValueError(
                    "attention_backend and attention_config.backend "
                    "are mutually exclusive"
                )
            # 复用验证器来处理 "auto" 和字符串到枚举的转换
            attention_config.backend = AttentionConfig.validate_backend_before(
                self.attention_backend  # 设置注意力后端
            )

        # 内核配置覆盖处理
        kernel_config = copy.deepcopy(self.kernel_config)  # 深拷贝内核配置
        if self.enable_flashinfer_autotune is not None:  # 如果指定了 FlashInfer 自动调优
            if kernel_config.enable_flashinfer_autotune is not None:
                raise ValueError(
                    "enable_flashinfer_autotune and "
                    "kernel_config.enable_flashinfer_autotune "
                    "are mutually exclusive"
                )
            kernel_config.enable_flashinfer_autotune = self.enable_flashinfer_autotune  # 设置 FlashInfer 自动调优
        if self.moe_backend != "auto":  # 如果指定了非 auto 的 MoE 后端
            kernel_config.moe_backend = self.moe_backend  # 设置 MoE 后端

        load_config = self.create_load_config()  # 创建模型加载配置

        # 将推理解析器传递到结构化输出配置中
        if self.reasoning_parser:  # 如果指定了推理解析器
            self.structured_outputs_config.reasoning_parser = self.reasoning_parser  # 设置推理解析器

        if self.reasoning_parser_plugin:  # 如果指定了推理解析器插件
            self.structured_outputs_config.reasoning_parser_plugin = (  # 设置推理解析器插件
                self.reasoning_parser_plugin
            )

        # 创建可观测性配置
        observability_config = ObservabilityConfig(
            show_hidden_metrics_for_version=self.show_hidden_metrics_for_version,  # 显示隐藏指标的版本
            otlp_traces_endpoint=self.otlp_traces_endpoint,  # OTLP 追踪端点
            collect_detailed_traces=self.collect_detailed_traces,  # 详细追踪模块
            kv_cache_metrics=self.kv_cache_metrics,  # KV 缓存指标
            kv_cache_metrics_sample=self.kv_cache_metrics_sample,  # KV 缓存指标采样率
            cudagraph_metrics=self.cudagraph_metrics,  # CUDAGraph 指标
            enable_layerwise_nvtx_tracing=self.enable_layerwise_nvtx_tracing,  # 逐层 NVTX 追踪
            enable_mfu_metrics=self.enable_mfu_metrics,  # MFU 指标
            enable_mm_processor_stats=self.enable_mm_processor_stats,  # 多模态处理器统计
            enable_logging_iteration_details=self.enable_logging_iteration_details,  # 迭代详情日志
        )

        # 编译配置覆盖处理
        compilation_config = copy.deepcopy(self.compilation_config)  # 深拷贝编译配置
        if self.cudagraph_capture_sizes is not None:  # 如果指定了 CUDAGraph 捕获大小
            if compilation_config.cudagraph_capture_sizes is not None:
                raise ValueError(
                    "cudagraph_capture_sizes and compilation_config."
                    "cudagraph_capture_sizes are mutually exclusive"
                )
            compilation_config.cudagraph_capture_sizes = self.cudagraph_capture_sizes  # 设置 CUDAGraph 捕获大小
        if self.max_cudagraph_capture_size is not None:  # 如果指定了最大 CUDAGraph 捕获大小
            if compilation_config.max_cudagraph_capture_size is not None:
                raise ValueError(
                    "max_cudagraph_capture_size and compilation_config."
                    "max_cudagraph_capture_size are mutually exclusive"
                )
            compilation_config.max_cudagraph_capture_size = (
                self.max_cudagraph_capture_size
            )

        # 创建卸载配置
        offload_config = OffloadConfig(
            offload_backend=self.offload_backend,  # 卸载后端
            uva=UVAOffloadConfig(  # UVA 卸载配置
                cpu_offload_gb=self.cpu_offload_gb,  # CPU 卸载大小（GB）
                cpu_offload_params=self.cpu_offload_params,  # CPU 卸载参数集合
            ),
            prefetch=PrefetchOffloadConfig(  # 预取卸载配置
                offload_group_size=self.offload_group_size,  # 卸载组大小
                offload_num_in_group=self.offload_num_in_group,  # 卸载组内数量
                offload_prefetch_step=self.offload_prefetch_step,  # 预取步数
                offload_params=self.offload_params,  # 卸载参数集合
            ),
        )

        # 构造并返回 VllmConfig 总配置对象
        config = VllmConfig(
            model_config=model_config,  # 模型配置
            cache_config=cache_config,  # 缓存配置
            parallel_config=parallel_config,  # 并行配置
            scheduler_config=scheduler_config,  # 调度器配置
            device_config=device_config,  # 设备配置
            load_config=load_config,  # 加载配置
            offload_config=offload_config,  # 卸载配置
            attention_config=attention_config,  # 注意力配置
            kernel_config=kernel_config,  # 内核配置
            lora_config=lora_config,  # LoRA 配置
            speculative_config=speculative_config,  # 推测解码配置
            structured_outputs_config=self.structured_outputs_config,  # 结构化输出配置
            observability_config=observability_config,  # 可观测性配置
            compilation_config=compilation_config,  # 编译优化配置
            kv_transfer_config=self.kv_transfer_config,  # KV 传输配置
            kv_events_config=self.kv_events_config,  # KV 事件配置
            ec_transfer_config=self.ec_transfer_config,  # EC 传输配置
            profiler_config=self.profiler_config,  # 性能分析器配置
            additional_config=self.additional_config,  # 额外自定义配置
            optimization_level=self.optimization_level,  # 优化级别
            performance_mode=self.performance_mode,  # 性能模式
            weight_transfer_config=self.weight_transfer_config,  # 权重传输配置
            shutdown_timeout=self.shutdown_timeout,  # 关机超时时间
        )

        return config  # 返回完整的 vLLM 配置

    def _check_feature_supported(self):
        """检查所请求的功能是否被当前平台支持，不支持则抛出错误。"""
        # 目前不支持并发部分预填充
        if (
            self.max_num_partial_prefills != SchedulerConfig.max_num_partial_prefills
            or self.max_long_partial_prefills
            != SchedulerConfig.max_long_partial_prefills
        ):
            _raise_unsupported_error(feature_name="Concurrent Partial Prefill")  # 抛出不支持错误

        if self.pipeline_parallel_size > 1:  # 如果启用了流水线并行
            supports_pp = getattr(  # 检查执行器后端是否支持流水线并行
                self.distributed_executor_backend, "supports_pp", False
            )
            if not supports_pp and self.distributed_executor_backend not in (  # 如果后端不支持且不是已知后端
                ParallelConfig.distributed_executor_backend,
                "ray",
                "mp",
                "external_launcher",
            ):
                name = (
                    "Pipeline Parallelism without Ray distributed "
                    "executor or multiprocessing executor or external "
                    "launcher"
                )
                _raise_unsupported_error(feature_name=name)

    @classmethod  # 类方法装饰器
    def get_batch_defaults(
        cls,
        world_size: int,
    ) -> tuple[dict[UsageContext | None, int], dict[UsageContext | None, int]]:
        """根据硬件和使用场景获取默认的批处理参数。

        根据设备类型（GPU 型号、TPU、CPU 等）和使用场景（LLM 类或 API 服务器），
        返回合适的 max_num_batched_tokens 和 max_num_seqs 默认值。

        参数:
            world_size: 总的并行世界大小（流水线并行 × 张量并行）
        返回:
            元组：(max_num_batched_tokens 默认值字典, max_num_seqs 默认值字典)
        """
        from vllm.usage.usage_lib import UsageContext  # 导入使用场景枚举

        default_max_num_batched_tokens: dict[UsageContext | None, int]  # 默认最大批处理 token 数
        default_max_num_seqs: dict[UsageContext | None, int]  # 默认最大序列数

        # 当没有用户覆盖时，根据使用场景设置默认值。
        # 对不同硬件使用不同的默认值。

        # 尝试查询当前平台的设备名称。如果失败，
        # 可能是因为导入 vLLM 的平台与实际运行 vLLM 的平台不同
        #（例如通过 Ray 扩展 vLLM 的情况），且没有 GPU。
        # 此时使用非 H100/H200 GPU 的默认值。
        try:
            device_memory = current_platform.get_device_total_memory()  # 获取设备总内存
            device_name = current_platform.get_device_name().lower()  # 获取设备名称（小写）
        except Exception:
            # 这仅用于设置 default_max_num_batched_tokens
            device_memory = 0  # 默认内存为 0
            device_name = ""  # 默认设备名为空

        # 注意(Kuntai)：对 A100 设置过大的 max_num_batched_tokens 会降低吞吐量，
        # 详见 PR #17885。因此这里额外检查设备名称以防止此类性能回退。
        if device_memory >= 70 * GiB_bytes and "a100" not in device_name:  # 大于 70GB 内存且非 A100
            # 对于 H100 和 MI300x 等高端 GPU，使用更大的默认值
            default_max_num_batched_tokens = {
                UsageContext.LLM_CLASS: 16384,
                UsageContext.OPENAI_API_SERVER: 8192,
            }
            default_max_num_seqs = {
                UsageContext.LLM_CLASS: 1024,
                UsageContext.OPENAI_API_SERVER: 1024,
            }
        else:  # 其他硬件
            # TODO(woosuk): 为其他硬件调优默认值
            default_max_num_batched_tokens = {
                UsageContext.LLM_CLASS: 8192,
                UsageContext.OPENAI_API_SERVER: 2048,
            }
            default_max_num_seqs = {
                UsageContext.LLM_CLASS: 256,
                UsageContext.OPENAI_API_SERVER: 256,
            }

        # TPU 特定的默认值
        if current_platform.is_tpu():  # 如果是 TPU 平台
            chip_name = current_platform.get_device_name()  # 获取 TPU 芯片名称

            if chip_name == "V6E":
                default_max_num_batched_tokens = {
                    UsageContext.LLM_CLASS: 2048,
                    UsageContext.OPENAI_API_SERVER: 1024,
                }
            elif chip_name == "V5E":
                default_max_num_batched_tokens = {
                    UsageContext.LLM_CLASS: 1024,
                    UsageContext.OPENAI_API_SERVER: 512,
                }
            elif chip_name == "V5P":
                default_max_num_batched_tokens = {
                    UsageContext.LLM_CLASS: 512,
                    UsageContext.OPENAI_API_SERVER: 256,
                }

        # CPU 特定的默认值
        if current_platform.is_cpu():  # 如果是 CPU 平台
            default_max_num_batched_tokens = {
                UsageContext.LLM_CLASS: 4096 * world_size,
                UsageContext.OPENAI_API_SERVER: 2048 * world_size,
            }
            default_max_num_seqs = {
                UsageContext.LLM_CLASS: 256 * world_size,
                UsageContext.OPENAI_API_SERVER: 128 * world_size,
            }

        return default_max_num_batched_tokens, default_max_num_seqs  # 返回默认的批处理参数

    def _set_default_chunked_prefill_and_prefix_caching_args(
        self, model_config: ModelConfig
    ) -> None:
        """设置分块预填充和前缀缓存的默认参数值。

        根据模型是否支持分块预填充和前缀缓存来设置默认值，
        并在用户选择的设置与模型推荐设置不匹配时发出警告。

        参数:
            model_config: 模型配置对象
        """
        default_chunked_prefill = model_config.is_chunked_prefill_supported  # 模型是否支持分块预填充
        default_prefix_caching = model_config.is_prefix_caching_supported  # 模型是否支持前缀缓存

        if self.enable_chunked_prefill is None:  # 如果用户未指定分块预填充设置
            self.enable_chunked_prefill = default_chunked_prefill

            logger.debug(
                "%s chunked prefill by default",
                "Enabling" if default_chunked_prefill else "Disabling",
            )
        elif (
            model_config.runner_type == "generate"
            and not self.enable_chunked_prefill
            and default_chunked_prefill
        ):
            logger.warning_once(
                "This model does not officially support disabling chunked prefill. "
                "Disabling this manually may cause the engine to crash "
                "or produce incorrect outputs.",
                scope="local",
            )
        elif (
            model_config.runner_type == "pooling"
            and self.enable_chunked_prefill
            and not default_chunked_prefill
        ):
            logger.warning_once(
                "This model does not officially support chunked prefill. "
                "Enabling this manually may cause the engine to crash "
                "or produce incorrect outputs.",
                scope="local",
            )

        if self.enable_prefix_caching is None:  # 如果用户未指定前缀缓存设置
            self.enable_prefix_caching = default_prefix_caching  # 使用模型的默认设置

            logger.debug(
                "%s prefix caching by default",
                "Enabling" if default_prefix_caching else "Disabling",
            )
        elif (
            model_config.runner_type == "pooling"
            and self.enable_prefix_caching
            and not default_prefix_caching
        ):
            logger.warning_once(
                "This model does not officially support prefix caching. "
                "Enabling this manually may cause the engine to crash "
                "or produce incorrect outputs.",
                scope="local",
            )

        # 在以下情况下禁用分块预填充和前缀缓存：
        # V1 版本中的 RISC-V CPU
        if current_platform.is_cpu() and current_platform.get_cpu_architecture() in (  # 如果是 RISC-V CPU
            CpuArchEnum.RISCV,
        ):
            logger.info(
                "Chunked prefill is not supported for"
                "RISC-V CPUs; "
                "disabling it for V1 backend."
            )
            self.enable_chunked_prefill = False
            logger.info(
                "Prefix caching is not supported for "
                "RISC-V CPUs; "
                "disabling it for V1 backend."
            )
            self.enable_prefix_caching = False

    def _set_default_max_num_seqs_and_batched_tokens_args(
        self,
        usage_context: UsageContext | None,
        model_config: ModelConfig,
    ):
        """设置最大序列数和最大批处理 token 数的默认值。

        根据使用场景、硬件类型和性能模式，自动计算合适的批处理参数。
        如果用户未指定这些值，则使用平台特定的默认值。

        参数:
            usage_context: 使用场景上下文
            model_config: 模型配置对象
        """
        world_size = self.pipeline_parallel_size * self.tensor_parallel_size  # 计算 world_size
        (
            default_max_num_batched_tokens,  # 默认最大批处理 token 数
            default_max_num_seqs,  # 默认最大序列数
        ) = self.get_batch_defaults(world_size)  # 获取默认批处理参数

        orig_max_num_batched_tokens = self.max_num_batched_tokens  # 保存原始值以判断是否为用户指定
        orig_max_num_seqs = self.max_num_seqs  # 保存原始值

        if self.max_num_batched_tokens is None:  # 如果用户未指定最大批处理 token 数
            self.max_num_batched_tokens = default_max_num_batched_tokens.get(
                usage_context,
                SchedulerConfig.DEFAULT_MAX_NUM_BATCHED_TOKENS,
            )

        if self.max_num_seqs is None:  # 如果用户未指定最大序列数
            self.max_num_seqs = default_max_num_seqs.get(  # 从默认值中获取
                usage_context,
                SchedulerConfig.DEFAULT_MAX_NUM_SEQS,
            )

        # 如果设置了吞吐量模式，将 max_num_batched_tokens 和 max_num_seqs 翻倍
        if self.performance_mode == "throughput":  # 吞吐量优先模式
            if orig_max_num_batched_tokens is None:  # 仅在用户未指定时翻倍
                self.max_num_batched_tokens *= 2  # 翻倍最大批处理 token 数
            if orig_max_num_seqs is None:  # 仅在用户未指定时翻倍
                self.max_num_seqs *= 2  # 翻倍最大序列数

        if orig_max_num_batched_tokens is None:  # 如果使用的是默认值（非用户指定）
            assert model_config.max_model_len is not None, (
                "max_model_len must be set by this point"
            )
            if not self.enable_chunked_prefill:  # 如果未启用分块预填充
                # 如果 max_model_len 太短，使用默认值以获得更高吞吐量
                self.max_num_batched_tokens = max(
                    model_config.max_model_len,
                    self.max_num_batched_tokens,
                )

            # 使用默认设置时，确保 max_num_batched_tokens 不超过模型限制。
            # 某些模型（如 Whisper）的嵌入与最大长度绑定。
            self.max_num_batched_tokens = min(  # 取较小值
                self.max_num_seqs * model_config.max_model_len,
                self.max_num_batched_tokens,
            )

            logger.debug(
                "Defaulting max_num_batched_tokens to %d for %s usage context.",
                self.max_num_batched_tokens,
                usage_context.value if usage_context else None,
            )

        if orig_max_num_seqs is None:  # 如果使用的是默认序列数
            assert self.max_num_batched_tokens is not None  # 类型检查断言
            self.max_num_seqs = min(self.max_num_seqs, self.max_num_batched_tokens)  # 确保序列数不超过批处理 token 数

            logger.debug(
                "Defaulting max_num_seqs to %d for %s usage context.",
                self.max_num_seqs,
                usage_context.value if usage_context else None,
            )


@dataclass  # 数据类装饰器
class AsyncEngineArgs(EngineArgs):
    """异步 vLLM 引擎参数类。

    继承自 EngineArgs，增加了异步引擎特有的参数，
    如请求日志记录功能。
    """

    enable_log_requests: bool = False  # 是否启用请求日志记录

    @staticmethod  # 静态方法装饰器
    def add_cli_args(
        parser: FlexibleArgumentParser, async_args_only: bool = False
    ) -> FlexibleArgumentParser:
        """添加异步引擎的命令行参数。

        参数:
            parser: 命令行参数解析器
            async_args_only: 如果为 True，仅添加异步特有的参数
        返回:
            添加了参数的解析器
        """
        # 初始化插件以更新解析器。例如，插件可能会向 --quantization 参数
        # 添加新的量化方法，或向 --device 参数添加新的设备类型。
        load_general_plugins()  # 加载通用插件
        if not async_args_only:  # 如果不是仅添加异步参数
            parser = EngineArgs.add_cli_args(parser)  # 先添加基础引擎参数
        parser.add_argument(  # 添加请求日志参数
            "--enable-log-requests",
            action=argparse.BooleanOptionalAction,  # 支持 --enable-log-requests 和 --no-enable-log-requests
            default=AsyncEngineArgs.enable_log_requests,  # 默认不启用
            help="Enable logging request information, dependent on log level:\n"  # 帮助文本
            "- INFO: Request ID, parameters and LoRA request.\n"
            "- DEBUG: Prompt inputs (e.g: text, token IDs).\n"
            "You can set the minimum log level via `VLLM_LOGGING_LEVEL`.",
        )
        current_platform.pre_register_and_update(parser)  # 让平台有机会更新解析器（添加平台特定参数）
        return parser  # 返回更新后的解析器


def _raise_unsupported_error(feature_name: str):
    """抛出功能不支持的错误。

    参数:
        feature_name: 不支持的功能名称
    抛出:
        NotImplementedError: 包含功能名称和建议的错误信息
    """
    msg = (
        f"{feature_name} is not supported. We recommend to "
        f"remove {feature_name} from your config."
    )
    raise NotImplementedError(msg)  # 抛出未实现错误


def human_readable_int(value: str) -> int:
    """解析人类可读的整数格式，如 '1k'、'2M' 等。
    支持小写（十进制）和大写（二进制）后缀，以及带小数点的十进制值。

    小写后缀使用十进制倍数：k=10^3, m=10^6, g=10^9, t=10^12
    大写后缀使用二进制倍数：K=2^10, M=2^20, G=2^30, T=2^40

    示例:
    - '1k' -> 1,000（十进制千）
    - '1K' -> 1,024（二进制千）
    - '25.6k' -> 25,600（十进制，支持小数）

    参数:
        value: 要解析的字符串
    返回:
        解析后的整数值
    """
    value = value.strip()  # 去除首尾空白

    match = re.fullmatch(r"(\d+(?:\.\d+)?)([kKmMgGtT])", value)  # 匹配数字+后缀格式
    if match:  # 如果匹配成功
        decimal_multiplier = {  # 十进制倍数（小写后缀）
            "k": 10**3,   # 千
            "m": 10**6,   # 百万
            "g": 10**9,   # 十亿
            "t": 10**12,  # 万亿
        }
        binary_multiplier = {  # 二进制倍数（大写后缀）
            "K": 2**10,   # KiB
            "M": 2**20,   # MiB
            "G": 2**30,   # GiB
            "T": 2**40,   # TiB
        }

        number, suffix = match.groups()  # 提取数字和后缀
        if suffix in decimal_multiplier:  # 如果是十进制后缀
            mult = decimal_multiplier[suffix]  # 获取倍数
            return int(float(number) * mult)  # 支持小数（如 25.6k）
        elif suffix in binary_multiplier:  # 如果是二进制后缀
            mult = binary_multiplier[suffix]  # 获取倍数
            # 二进制后缀不允许使用小数
            try:
                return int(number) * mult  # 转换为整数并乘以倍数
            except ValueError as e:  # 如果数字包含小数点，int() 会失败
                raise argparse.ArgumentTypeError(
                    "Decimals are not allowed "
                    f"with binary suffixes like {suffix}. Did you mean to use "
                    f"{number}{suffix.lower()} instead?"  # 建议使用对应的小写后缀
                ) from e

    # 普通数字，直接转换
    return int(value)  # 将字符串转换为整数


def human_readable_int_or_auto(value: str) -> int:
    """解析人类可读的整数格式，同时支持 'auto' 或 '-1' 表示自动检测。

    扩展了 human_readable_int 的功能，额外接受 -1 或 'auto' 作为特殊值。

    示例:
    - '1k' -> 1,000
    - '1K' -> 1,024
    - '25.6k' -> 25,600
    - '-1' 或 'auto' -> -1（表示自动检测的特殊值）

    参数:
        value: 要解析的字符串
    返回:
        解析后的整数值，-1 表示自动检测
    """
    value = value.strip()  # 去除首尾空白

    if value == "-1" or value.lower() == "auto":  # 如果是自动检测的特殊值
        return -1  # 返回 -1 表示自动检测

    return human_readable_int(value)  # 否则使用标准的人类可读整数解析
