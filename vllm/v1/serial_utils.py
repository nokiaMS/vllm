# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import importlib
import pickle
from collections.abc import Callable, Sequence
from functools import partial
from inspect import isclass
from types import FunctionType
from typing import Any, TypeAlias, get_type_hints

import cloudpickle
import msgspec
import numpy as np
import torch
import zmq
from msgspec import msgpack
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

from vllm import envs
from vllm.logger import init_logger
from vllm.multimodal.inputs import (
    BaseMultiModalField,
    MultiModalBatchedField,
    MultiModalFieldConfig,
    MultiModalFieldElem,
    MultiModalFlatField,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    MultiModalSharedField,
    NestedTensors,
)
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.utils import tensor_data

logger = init_logger(__name__)

# [中文注释] msgpack 扩展类型编号，用于标识自定义序列化的数据类型：
#   PICKLE=1: 标准 pickle 序列化的对象
#   CLOUDPICKLE=2: cloudpickle 序列化的对象（支持闭包/lambda）
#   RAW_VIEW=3: 原始内存视图（用于内联的小 tensor/ndarray 数据）
CUSTOM_TYPE_PICKLE = 1
CUSTOM_TYPE_CLOUDPICKLE = 2
CUSTOM_TYPE_RAW_VIEW = 3

# MultiModalField class serialization type map.
# These need to list all possible field types and match them
# to factory methods in `MultiModalFieldConfig`.
# [中文注释] 多模态字段类 → 工厂方法名的映射表。
# 序列化时记录工厂方法名，反序列化时通过 MultiModalFieldConfig 的对应工厂方法重建字段。
MMF_CLASS_TO_FACTORY: dict[type[BaseMultiModalField], str] = {
    MultiModalFlatField: "flat",
    MultiModalSharedField: "shared",
    MultiModalBatchedField: "batched",
}

# [中文注释] 字节类数据的联合类型别名，涵盖 ZMQ 传输中可能出现的所有字节容器类型
bytestr: TypeAlias = bytes | bytearray | memoryview | zmq.Frame


# [中文注释] 记录不安全序列化警告（仅打印一次）
def _log_insecure_serialization_warning():
    logger.warning_once(
        "Allowing insecure serialization using pickle due to "
        "VLLM_ALLOW_INSECURE_SERIALIZATION=1"
    )


# [中文注释] 获取对象的类型字符串表示，返回 (模块名, 类名) 元组，用于序列化时保留类型信息
def _typestr(val: Any) -> tuple[str, str] | None:
    if val is None:
        return None
    t = type(val)
    return t.__module__, t.__qualname__


# [中文注释] 递归编码嵌套 list/dict 结构的类型信息。
# 将每个叶子节点替换为 (模块名, 类名) 元组，用于反序列化时恢复正确的类型。
def _encode_type_info_recursive(obj: Any) -> Any:
    """Recursively encode type information for nested structures of
    lists/dicts."""
    if obj is None:
        return None
    if type(obj) is list:
        return [_encode_type_info_recursive(item) for item in obj]
    if type(obj) is dict:
        return {k: _encode_type_info_recursive(v) for k, v in obj.items()}
    return _typestr(obj)


# [中文注释] 递归解码嵌套 list/dict 结构的类型信息。
# 根据编码时记录的类型信息，使用 convert_fn 将原始数据还原为正确的 Python 类型。
def _decode_type_info_recursive(
    type_info: Any, data: Any, convert_fn: Callable[[Sequence[str], Any], Any]
) -> Any:
    """Recursively decode type information for nested structures of
    lists/dicts."""
    if type_info is None:
        return data
    if isinstance(type_info, dict):
        assert isinstance(data, dict)
        return {
            k: _decode_type_info_recursive(type_info[k], data[k], convert_fn)
            for k in type_info
        }
    if isinstance(type_info, list) and (
        # Exclude serialized tensors/numpy arrays.
        len(type_info) != 2 or not isinstance(type_info[0], str)
    ):
        assert isinstance(data, list)
        return [
            _decode_type_info_recursive(ti, d, convert_fn)
            for ti, d in zip(type_info, data)
        ]
    return convert_fn(type_info, data)


# [中文注释] RPC 工具方法返回值的包装类。
# 序列化/反序列化时需要特殊处理，因为工具方法的返回值类型不固定。
class UtilityResult:
    """Wrapper for special handling when serializing/deserializing."""

    def __init__(self, r: Any = None):
        self.result = r


# [中文注释] 自定义 msgpack 编码器，支持 torch.Tensor 和 numpy.ndarray 的高效序列化。
# 核心设计：
#   - 小于阈值（默认 256B）的数组内联到 msgpack 主缓冲区中
#   - 大于阈值的数组通过 aux_buffers 收集其原始内存指针，
#     作为 ZMQ 多帧消息的独立帧发送，实现零拷贝传输
#   - 编码结果为 [msgpack主数据, tensor缓冲区1, tensor缓冲区2, ...] 的列表
class MsgpackEncoder:
    """Encoder with custom torch tensor and numpy array serialization.

    Note that unlike vanilla `msgspec` Encoders, this interface is generally
    not thread-safe when encoding tensors / numpy arrays.

    By default, arrays below 256B are serialized inline Larger will get sent
    via dedicated messages. Note that this is a per-tensor limit.
    """

    def __init__(self, size_threshold: int | None = None):
        if size_threshold is None:
            # [中文注释] 零拷贝的大小阈值，低于此值的 tensor 内联序列化
            size_threshold = envs.VLLM_MSGPACK_ZERO_COPY_THRESHOLD
        # [中文注释] 注册自定义编码钩子 enc_hook，处理 msgspec 原生不支持的类型
        self.encoder = msgpack.Encoder(enc_hook=self.enc_hook)
        # This is used as a local stash of buffers that we can then access from
        # our custom `msgspec` hook, `enc_hook`. We don't have a way to
        # pass custom data to the hook otherwise.
        # [中文注释] aux_buffers 用于在 enc_hook 回调中暂存大 tensor 的内存引用。
        # 由于 msgspec 的 hook 机制不支持传递额外参数，
        # 所以通过实例变量作为 hook 和 encode 方法之间的通信桥梁。
        self.aux_buffers: list[bytestr] | None = None
        self.size_threshold = size_threshold
        if envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
            _log_insecure_serialization_warning()

    # [中文注释] 将对象编码为 msgpack 格式。返回值是一个缓冲区列表：
    #   bufs[0] = msgpack 主数据（包含结构和小 tensor 的内联数据）
    #   bufs[1:] = 大 tensor/ndarray 的原始内存引用（零拷贝）
    # 这个列表会直接作为 ZMQ send_multipart 的多帧消息发送。
    def encode(self, obj: Any) -> Sequence[bytestr]:
        try:
            self.aux_buffers = bufs = [b""]
            bufs[0] = self.encoder.encode(obj)
            # This `bufs` list allows us to collect direct pointers to backing
            # buffers of tensors and np arrays, and return them along with the
            # top-level encoded buffer instead of copying their data into the
            # new buffer.
            return bufs
        finally:
            self.aux_buffers = None

    # [中文注释] 将对象编码到预分配的 bytearray 中（用于缓冲区复用，减少内存分配）。
    # Output IO 线程使用此方法，通过复用 bytearray 降低 GC 压力。
    def encode_into(self, obj: Any, buf: bytearray) -> Sequence[bytestr]:
        try:
            self.aux_buffers = [buf]
            bufs = self.aux_buffers
            self.encoder.encode_into(obj, buf)
            return bufs
        finally:
            self.aux_buffers = None

    # [中文注释] 自定义编码钩子：当 msgspec 遇到不认识的类型时调用此方法进行转换。
    # 支持的类型：torch.Tensor, numpy.ndarray, slice, 多模态数据, UtilityResult 等。
    # 不支持的类型在允许不安全序列化时回退到 pickle/cloudpickle。
    def enc_hook(self, obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return self._encode_tensor(obj)

        # Fall back to pickle for object or void kind ndarrays.
        # [中文注释] 对于 object 或 void 类型的 ndarray，回退到 pickle（无法用 frombuffer 重建）
        if isinstance(obj, np.ndarray) and obj.dtype.kind not in ("O", "V"):
            return self._encode_ndarray(obj)

        if isinstance(obj, slice):
            # We are assuming only int-based values will be used here.
            return tuple(
                int(v) if v is not None else None
                for v in (obj.start, obj.stop, obj.step)
            )

        # [中文注释] 多模态输入数据的序列化
        if isinstance(obj, MultiModalKwargsItem):
            return self._encode_mm_item(obj)

        if isinstance(obj, MultiModalKwargsItems):
            return self._encode_mm_items(obj)

        # [中文注释] RPC 工具方法返回值的序列化
        if isinstance(obj, UtilityResult):
            result = obj.result
            if not envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
                return None, result
            # Since utility results are not strongly typed, we recursively
            # encode type information for nested structures of lists/dicts
            # to help with correct msgspec deserialization.
            return _encode_type_info_recursive(result), result

        # [中文注释] 以下为不安全序列化的回退路径
        if not envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
            raise TypeError(
                f"Object of type {type(obj)} is not serializable"
                "Set VLLM_ALLOW_INSECURE_SERIALIZATION=1 to allow "
                "fallback to pickle-based serialization."
            )

        # [中文注释] 函数对象使用 cloudpickle（支持闭包），pickle 对方法序列化有问题
        if isinstance(obj, FunctionType):
            # `pickle` is generally faster than cloudpickle, but can have
            # problems serializing methods.
            return msgpack.Ext(CUSTOM_TYPE_CLOUDPICKLE, cloudpickle.dumps(obj))

        # [中文注释] 其他未知类型使用标准 pickle
        return msgpack.Ext(
            CUSTOM_TYPE_PICKLE, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        )

    # [中文注释] 编码 numpy 数组。序列化格式为 (dtype字符串, shape元组, 数据)。
    # 数据部分根据大小分为两种策略：
    #   - 小数组：内联为 msgpack 扩展类型 RAW_VIEW
    #   - 大数组：记录在 aux_buffers 中的索引，实现零拷贝
    def _encode_ndarray(
        self, obj: np.ndarray
    ) -> tuple[str, tuple[int, ...], int | memoryview]:
        assert self.aux_buffers is not None
        # If the array is non-contiguous, we need to copy it first
        arr_data = obj.data if obj.flags.c_contiguous else obj.tobytes()
        if not obj.shape or obj.nbytes < self.size_threshold:
            # Encode small arrays and scalars inline. Using this extension type
            # ensures we can avoid copying when decoding.
            data = msgpack.Ext(CUSTOM_TYPE_RAW_VIEW, arr_data)
        else:
            # Otherwise encode index of backing buffer to avoid copy.
            data = len(self.aux_buffers)
            self.aux_buffers.append(arr_data)

        # We serialize the ndarray as a tuple of native types.
        # The data is either inlined if small, or an index into a list of
        # backing buffers that we've stashed in `aux_buffers`.
        return obj.dtype.str, obj.shape, data

    # [中文注释] 编码 torch.Tensor。序列化格式为 (dtype字符串, shape元组, 数据)。
    # 与 ndarray 编码逻辑类似，同样采用大小阈值区分内联/零拷贝策略。
    def _encode_tensor(
        self, obj: torch.Tensor
    ) -> tuple[str, tuple[int, ...], int | memoryview]:
        assert self.aux_buffers is not None
        # view the tensor as a contiguous 1D array of bytes
        arr_data = tensor_data(obj)
        if obj.nbytes < self.size_threshold:
            # Smaller tensors are encoded inline, just like ndarrays.
            data = msgpack.Ext(CUSTOM_TYPE_RAW_VIEW, arr_data)
        else:
            # Otherwise encode index of backing buffer to avoid copy.
            data = len(self.aux_buffers)
            self.aux_buffers.append(arr_data)
        # [中文注释] 去掉 "torch." 前缀，如 "torch.float16" → "float16"
        dtype = str(obj.dtype).removeprefix("torch.")
        return dtype, obj.shape, data

    # ==================== 多模态数据编码方法 ====================

    # [中文注释] 编码多模态数据集合，按模态名称（如 "image", "audio"）分组
    def _encode_mm_items(self, items: MultiModalKwargsItems) -> dict[str, Any]:
        return {
            modality: [self._encode_mm_item(item) for item in itemlist]
            for modality, itemlist in items.items()
        }

    # [中文注释] 编码单个多模态数据项，包含多个字段元素
    def _encode_mm_item(self, item: MultiModalKwargsItem) -> dict[str, Any]:
        return {key: self._encode_mm_field_elem(elem) for key, elem in item.items()}

    # [中文注释] 编码多模态字段元素，包含 data（嵌套 tensor）和 field（字段处理器）两部分
    def _encode_mm_field_elem(self, elem: MultiModalFieldElem) -> dict[str, Any]:
        return {
            "data": (
                None if elem.data is None else self._encode_nested_tensors(elem.data)
            ),
            "field": self._encode_mm_field(elem.field),
        }

    # [中文注释] 递归编码嵌套 tensor 结构（tensor 列表的列表...）
    def _encode_nested_tensors(self, nt: NestedTensors) -> Any:
        if isinstance(nt, torch.Tensor):
            return self._encode_tensor(nt)
        if isinstance(nt, (int, float)):
            # Although it violates NestedTensors type, MultiModalKwargs
            # values are sometimes floats.
            return nt
        return [self._encode_nested_tensors(x) for x in nt]

    # [中文注释] 编码多模态字段处理器。记录工厂方法名和所有字段参数，
    # 反序列化时通过工厂方法重建字段对象。
    def _encode_mm_field(self, field: BaseMultiModalField):
        # Figure out the factory name for the field type.
        name = MMF_CLASS_TO_FACTORY.get(field.__class__)
        if not name:
            raise TypeError(f"Unsupported field type: {field.__class__}")

        # We just need to copy all of the field values in order
        # which will be then used to reconstruct the field.
        factory_kw = {f.name: getattr(field, f.name) for f in dataclasses.fields(field)}
        return name, factory_kw


# [中文注释] 自定义 msgpack 解码器，与 MsgpackEncoder 配对使用。
# 解码时根据 aux_buffers 中的数据帧还原零拷贝传输的大 tensor/ndarray。
class MsgpackDecoder:
    """Decoder with custom torch tensor and numpy array serialization.

    Note that unlike vanilla `msgspec` Decoders, this interface is generally
    not thread-safe when encoding tensors / numpy arrays.
    """

    def __init__(self, t: Any | None = None, share_mem: bool = True):
        # [中文注释] share_mem=True 时共享 ZMQ 接收缓冲区内存（零拷贝），
        # share_mem=False 时会 clone/pin_memory（用于需要独立内存的场景）
        self.share_mem = share_mem
        self.pin_tensors = is_pin_memory_available()
        args = () if t is None else (t,)
        # [中文注释] ext_hook：处理 msgpack 扩展类型（RAW_VIEW, PICKLE, CLOUDPICKLE）
        # dec_hook：处理自定义 Python 类型（Tensor, ndarray, slice 等）
        self.decoder = msgpack.Decoder(
            *args, ext_hook=self.ext_hook, dec_hook=self.dec_hook
        )
        # [中文注释] 存储 ZMQ 多帧消息中的辅助缓冲区帧（对应编码器的 aux_buffers）
        self.aux_buffers: Sequence[bytestr] = ()
        if envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
            _log_insecure_serialization_warning()

    # [中文注释] 解码 msgpack 数据。输入可以是单个字节缓冲区，或 ZMQ 多帧消息列表：
    #   bufs[0] = msgpack 主数据
    #   bufs[1:] = 大 tensor/ndarray 的原始内存帧
    # 解码过程中 dec_hook 会通过 self.aux_buffers 访问这些辅助帧。
    def decode(self, bufs: bytestr | Sequence[bytestr]) -> Any:
        if isinstance(bufs, bytestr):  # type: ignore
            # [中文注释] 单帧消息，没有辅助缓冲区（所有 tensor 都是内联的）
            return self.decoder.decode(bufs)

        # [中文注释] 多帧消息，暂存辅助缓冲区供 dec_hook 使用
        self.aux_buffers = bufs
        try:
            return self.decoder.decode(bufs[0])
        finally:
            self.aux_buffers = ()

    # [中文注释] 自定义解码钩子：将 msgpack 原生类型转换为目标 Python 类型 t。
    # 根据目标类型分发到对应的解码方法。
    def dec_hook(self, t: type, obj: Any) -> Any:
        # Given native types in `obj`, convert to type `t`.
        if isclass(t):
            if issubclass(t, np.ndarray):
                return self._decode_ndarray(obj)
            if issubclass(t, torch.Tensor):
                return self._decode_tensor(obj)
            if t is slice:
                return slice(*obj)
            if issubclass(t, MultiModalKwargsItem):
                return self._decode_mm_item(obj)
            if issubclass(t, MultiModalKwargsItems):
                return self._decode_mm_items(obj)
            if t is UtilityResult:
                return self._decode_utility_result(obj)
        return obj

    # [中文注释] 解码 RPC 工具方法的返回值。
    # obj 格式为 (类型信息, 原始数据)，类型信息用于恢复正确的 Python 类型。
    def _decode_utility_result(self, obj: Any) -> UtilityResult:
        result_type, result = obj
        if result_type is not None:
            if not envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
                raise TypeError(
                    "VLLM_ALLOW_INSECURE_SERIALIZATION must "
                    "be set to use custom utility result types"
                )
            # Use recursive decoding to handle nested structures
            result = _decode_type_info_recursive(
                result_type, result, self._convert_result
            )
        return UtilityResult(result)

    # [中文注释] 根据 (模块名, 类名) 元组动态导入类型，并使用 msgspec.convert 进行类型转换
    def _convert_result(self, result_type: Sequence[str], result: Any) -> Any:
        if result_type is None:
            return result
        mod_name, name = result_type
        mod = importlib.import_module(mod_name)
        result_type = getattr(mod, name)
        return msgspec.convert(result, result_type, dec_hook=self.dec_hook)

    # [中文注释] 解码 numpy 数组。输入格式为 (dtype, shape, data)。
    # data 为整数索引时从 aux_buffers 取缓冲区（零拷贝），否则为内联的 RAW_VIEW。
    # 注意：零拷贝解码会锁定整个 ZMQ 接收缓冲区，因此数组不应被长期持有。
    def _decode_ndarray(self, arr: Any) -> np.ndarray:
        dtype, shape, data = arr
        # zero-copy decode. We assume the ndarray will not be kept around,
        # as it now locks the whole received message buffer in memory.
        buffer = self.aux_buffers[data] if isinstance(data, int) else data
        arr = np.frombuffer(buffer, dtype=dtype)
        if not self.share_mem:
            arr = arr.copy()
        return arr.reshape(shape)

    # [中文注释] 解码 torch.Tensor。输入格式为 (dtype, shape, data)。
    # 解码流程：
    #   1. 从 aux_buffers 或内联数据获取原始字节
    #   2. 用 torch.frombuffer 创建 uint8 视图（避免拷贝）
    #   3. 根据 share_mem 决定是否 clone/pin_memory
    #   4. 通过 view 转换为目标 dtype 和 shape
    def _decode_tensor(self, arr: Any) -> torch.Tensor:
        dtype, shape, data = arr
        is_aux = isinstance(data, int)
        buffer = self.aux_buffers[data] if is_aux else data
        buffer = buffer if isinstance(buffer, memoryview) else memoryview(buffer)
        torch_dtype = getattr(torch, dtype)
        assert isinstance(torch_dtype, torch.dtype)
        if not buffer.nbytes:  # torch.frombuffer doesn't like empty buffers
            assert 0 in shape
            return torch.empty(shape, dtype=torch_dtype)
        # Create uint8 array
        arr = torch.frombuffer(buffer, dtype=torch.uint8)
        # Clone ensures tensor is backed by pytorch-owned memory for safe
        # future async CPU->GPU transfer.
        # Pin larger tensors for more efficient CPU->GPU transfer.
        if not is_aux:
            # [中文注释] 内联数据：必须 clone（原始内存属于 msgpack 缓冲区）
            arr = arr.clone()
        elif not self.share_mem:
            # [中文注释] 非共享内存模式：pin_memory（如可用）或 clone
            arr = arr.pin_memory() if self.pin_tensors else arr.clone()
        # Convert back to proper shape & type
        return arr.view(torch_dtype).view(shape)

    # ==================== 多模态数据解码方法 ====================

    # [中文注释] 解码多模态数据集合，按模态名称分组重建
    def _decode_mm_items(self, obj: dict[str, Any]) -> MultiModalKwargsItems:
        return MultiModalKwargsItems(
            {
                modality: [self._decode_mm_item(item) for item in itemlist]
                for modality, itemlist in obj.items()
            }
        )

    # [中文注释] 解码单个多模态数据项
    def _decode_mm_item(self, obj: dict[str, Any]) -> MultiModalKwargsItem:
        return MultiModalKwargsItem(
            {key: self._decode_mm_field_elem(elem) for key, elem in obj.items()}
        )

    # [中文注释] 解码多模态字段元素：还原嵌套 tensor 数据，并通过工厂方法重建字段处理器
    def _decode_mm_field_elem(self, obj: dict[str, Any]) -> MultiModalFieldElem:
        if obj["data"] is not None:
            obj["data"] = self._decode_nested_tensors(obj["data"])

        # Reconstruct the field processor using MultiModalFieldConfig
        factory_meth_name, factory_kw = obj["field"]
        factory_meth = getattr(MultiModalFieldConfig, factory_meth_name)

        # Special case: decode the union "slices" field of
        # MultiModalFlatField
        if factory_meth_name == "flat":
            factory_kw["slices"] = self._decode_nested_slices(factory_kw["slices"])

        obj["field"] = factory_meth("", **factory_kw).field
        return MultiModalFieldElem(**obj)

    # [中文注释] 递归解码嵌套 tensor 结构。
    # 通过首元素类型区分：首元素为字符串说明是 (dtype, shape, data) 格式的 tensor。
    def _decode_nested_tensors(self, obj: Any) -> NestedTensors:
        if isinstance(obj, (int, float)):
            # Although it violates NestedTensors type, MultiModalKwargs
            # values are sometimes floats.
            return obj
        if not isinstance(obj, list):
            raise TypeError(f"Unexpected NestedTensors contents: {type(obj)}")
        if obj and isinstance(obj[0], str):
            return self._decode_tensor(obj)
        return [self._decode_nested_tensors(x) for x in obj]

    # [中文注释] 递归将 (start, stop, step) 元组还原为 slice 对象
    def _decode_nested_slices(self, obj: Any) -> Any:
        assert isinstance(obj, (list, tuple))
        if obj and not isinstance(obj[0], (list, tuple)):
            return slice(*obj)
        return [self._decode_nested_slices(x) for x in obj]

    # [中文注释] msgpack 扩展类型解码钩子。处理三种自定义扩展类型：
    #   RAW_VIEW: 内联的小 tensor/ndarray 原始字节，直接返回 memoryview
    #   PICKLE: 标准 pickle 序列化的对象（需要启用不安全序列化）
    #   CLOUDPICKLE: cloudpickle 序列化的函数/闭包（需要启用不安全序列化）
    def ext_hook(self, code: int, data: memoryview) -> Any:
        if code == CUSTOM_TYPE_RAW_VIEW:
            return data

        if envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
            if code == CUSTOM_TYPE_PICKLE:
                return pickle.loads(data)
            if code == CUSTOM_TYPE_CLOUDPICKLE:
                return cloudpickle.loads(data)

        raise NotImplementedError(f"Extension type code {code} is not supported")


# [中文注释] 在对象上执行方法，支持三种方法指定方式：
#   str: 方法名字符串，通过 getattr 查找对象方法
#   bytes: cloudpickle 序列化的函数，反序列化后以 obj 为第一个参数调用
#   Callable: 直接调用的函数，以 obj 为第一个参数
# 用于 Executor 的远程方法调用（RPC）场景。
def run_method(
    obj: Any,
    method: str | bytes | Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """
    Run a method of an object with the given arguments and keyword arguments.
    If the method is string, it will be converted to a method using getattr.
    If the method is serialized bytes and will be deserialized using
    cloudpickle.
    If the method is a callable, it will be called directly.
    """
    if isinstance(method, bytes):
        func = partial(cloudpickle.loads(method), obj)
    elif isinstance(method, str):
        try:
            func = getattr(obj, method)
        except AttributeError:
            raise NotImplementedError(
                f"Method {method!r} is not implemented."
            ) from None
    else:
        func = partial(method, obj)  # type: ignore
    return func(*args, **kwargs)


# [中文注释] Pydantic 与 msgspec.Struct 的兼容层 Mixin。
# 使 msgspec.Struct 子类可以用于 Pydantic 模型验证（如 FastAPI 的请求/响应模型），
# 支持在 /docs（OpenAPI）中正确显示 schema，并保留字段默认值。
# Pydantic 会缓存此 schema，不会在每次验证时重复调用。
class PydanticMsgspecMixin:
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """
        Make msgspec.Struct compatible with Pydantic, respecting defaults.
        Handle JSON=>msgspec.Struct. Used when exposing msgspec.Struct to the
        API as input or in `/docs`. Note this is cached by Pydantic and not
        called on every validation.
        """
        msgspec_fields = {f.name: f for f in msgspec.structs.fields(source_type)}
        type_hints = get_type_hints(source_type)

        # Build the Pydantic typed_dict_field for each msgspec field
        fields = {}
        for name, hint in type_hints.items():
            msgspec_field = msgspec_fields[name]

            # typed_dict_field using the handler to get the schema
            field_schema = handler(hint)

            # Add default value to the schema.
            if msgspec_field.default_factory is not msgspec.NODEFAULT:
                wrapped_schema = core_schema.with_default_schema(
                    schema=field_schema,
                    default_factory=msgspec_field.default_factory,
                )
                fields[name] = core_schema.typed_dict_field(wrapped_schema)
            elif msgspec_field.default is not msgspec.NODEFAULT:
                wrapped_schema = core_schema.with_default_schema(
                    schema=field_schema,
                    default=msgspec_field.default,
                )
                fields[name] = core_schema.typed_dict_field(wrapped_schema)
            else:
                # No default, so Pydantic will treat it as required
                fields[name] = core_schema.typed_dict_field(field_schema)
        return core_schema.no_info_after_validator_function(
            cls._validate_msgspec,
            core_schema.typed_dict_schema(fields),
        )

    @classmethod
    def _validate_msgspec(cls, value: Any) -> Any:
        """Validate and convert input to msgspec.Struct instance."""
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        return msgspec.convert(value, type=cls)
