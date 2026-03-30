# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention backend registry"""

from collections.abc import Callable  # 导入Callable类型，用于函数类型注解
from enum import Enum, EnumMeta  # 导入枚举基类和枚举元类
from typing import TYPE_CHECKING, cast  # 导入类型检查标志和类型转换工具

from vllm.logger import init_logger  # 导入日志初始化工具
from vllm.utils.import_utils import resolve_obj_by_qualname  # 导入根据全限定名解析对象的工具函数

if TYPE_CHECKING:  # 仅在类型检查时导入，避免运行时循环依赖
    from vllm.v1.attention.backend import AttentionBackend  # 导入注意力后端基类（仅用于类型注解）

logger = init_logger(__name__)  # 初始化当前模块的日志记录器


class _AttentionBackendEnumMeta(EnumMeta):
    """注意力后端枚举的元类，用于在查找失败时提供更友好的错误信息。"""

    def __getitem__(cls, name: str):
        """根据名称获取后端枚举成员，查找失败时给出可用后端列表的提示信息。"""
        try:
            return super().__getitem__(name)  # 调用父类方法按名称查找枚举成员
        except KeyError:  # 如果名称不存在则捕获KeyError异常
            members = cast("dict[str, Enum]", cls.__members__).keys()  # 获取所有枚举成员的名称集合
            valid_backends = ", ".join(members)  # 将所有有效后端名称用逗号拼接成字符串
            raise ValueError(  # 抛出包含有效选项列表的ValueError异常
                f"Unknown attention backend: '{name}'. "
                f"Valid options are: {valid_backends}"
            ) from None  # 使用from None隐藏原始异常链


class AttentionBackendEnum(Enum, metaclass=_AttentionBackendEnumMeta):
    """注意力后端枚举类，列举所有支持的注意力后端。

    枚举值为默认的类路径字符串，可在运行时通过 register_backend() 覆盖。

    获取实际的后端类（会考虑覆盖）请使用：
        backend.get_class()
    """

    FLASH_ATTN = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"  # FlashAttention后端
    FLASH_ATTN_DIFFKV = (  # FlashAttention差异KV后端（支持不同的Key/Value维度）
        "vllm.v1.attention.backends.flash_attn_diffkv.FlashAttentionDiffKVBackend"
    )
    TRITON_ATTN = "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"  # Triton注意力后端
    ROCM_ATTN = "vllm.v1.attention.backends.rocm_attn.RocmAttentionBackend"  # ROCm平台注意力后端
    ROCM_AITER_MLA = "vllm.v1.attention.backends.mla.rocm_aiter_mla.AiterMLABackend"  # ROCm AITER MLA后端
    ROCM_AITER_TRITON_MLA = (  # ROCm AITER Triton MLA后端
        "vllm.v1.attention.backends.mla.aiter_triton_mla.AiterTritonMLABackend"
    )
    ROCM_AITER_FA = (  # ROCm AITER FlashAttention后端
        "vllm.v1.attention.backends.rocm_aiter_fa.AiterFlashAttentionBackend"
    )
    ROCM_AITER_MLA_SPARSE = (  # ROCm AITER MLA稀疏后端
        "vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse.ROCMAiterMLASparseBackend"
    )
    XPU_MLA_SPARSE = "vllm.v1.attention.backends.mla.xpu_mla_sparse.XPUMLASparseBackend"  # XPU平台MLA稀疏后端
    TORCH_SDPA = ""  # PyTorch SDPA后端，此标签仅用于视觉Transformer（ViT）
    FLASHINFER = "vllm.v1.attention.backends.flashinfer.FlashInferBackend"  # FlashInfer后端
    FLASHINFER_MLA = (  # FlashInfer MLA后端
        "vllm.v1.attention.backends.mla.flashinfer_mla.FlashInferMLABackend"
    )
    FLASHINFER_MLA_SPARSE = (  # FlashInfer MLA稀疏后端
        "vllm.v1.attention.backends.mla.flashinfer_mla_sparse."
        "FlashInferMLASparseBackend"
    )
    TRITON_MLA = "vllm.v1.attention.backends.mla.triton_mla.TritonMLABackend"  # Triton MLA后端
    CUTLASS_MLA = "vllm.v1.attention.backends.mla.cutlass_mla.CutlassMLABackend"  # CUTLASS MLA后端
    FLASHMLA = "vllm.v1.attention.backends.mla.flashmla.FlashMLABackend"  # FlashMLA后端
    FLASHMLA_SPARSE = (  # FlashMLA稀疏后端
        "vllm.v1.attention.backends.mla.flashmla_sparse.FlashMLASparseBackend"
    )
    FLASH_ATTN_MLA = "vllm.v1.attention.backends.mla.flashattn_mla.FlashAttnMLABackend"  # FlashAttention MLA后端
    NO_ATTENTION = "vllm.v1.attention.backends.no_attention.NoAttentionBackend"  # 无注意力后端（用于不需要注意力的场景）
    FLEX_ATTENTION = "vllm.v1.attention.backends.flex_attention.FlexAttentionBackend"  # Flex注意力后端
    TREE_ATTN = "vllm.v1.attention.backends.tree_attn.TreeAttentionBackend"  # 树形注意力后端
    ROCM_AITER_UNIFIED_ATTN = (  # ROCm AITER统一注意力后端
        "vllm.v1.attention.backends.rocm_aiter_unified_attn."
        "RocmAiterUnifiedAttentionBackend"
    )
    CPU_ATTN = "vllm.v1.attention.backends.cpu_attn.CPUAttentionBackend"  # CPU注意力后端
    # 第三方/自定义后端的占位符 - 使用前必须先注册
    # 设置为None以避免与其他值为空字符串的后端产生别名冲突
    CUSTOM = None  # 自定义后端占位符

    def get_path(self, include_classname: bool = True) -> str:
        """获取此后端的类路径（会考虑覆盖注册）。

        参数:
            include_classname: 是否包含类名，默认为True

        返回:
            完全限定的类路径字符串

        异常:
            ValueError: 如果 Backend.CUSTOM 未注册就使用时抛出
        """
        path = _ATTN_OVERRIDES.get(self, self.value)  # 优先从覆盖字典获取路径，否则使用默认枚举值
        if not path:  # 如果路径为空（未注册的自定义后端）
            raise ValueError(  # 抛出异常提示需要先注册
                f"Backend {self.name} must be registered before use. "
                f"Use register_backend(Backend.{self.name}, 'your.module.YourClass')"
            )
        if not include_classname:  # 如果不需要包含类名
            path = path.rsplit(".", 1)[0]  # 去掉最后一个点号后面的类名部分，只保留模块路径
        return path  # 返回类路径字符串

    def get_class(self) -> "type[AttentionBackend]":
        """获取后端类对象（会考虑覆盖注册）。

        返回:
            后端类对象

        异常:
            ImportError: 如果后端类无法导入时抛出
            ValueError: 如果 Backend.CUSTOM 未注册就使用时抛出
        """
        return resolve_obj_by_qualname(self.get_path())  # 根据全限定路径解析并返回对应的类对象

    def is_overridden(self) -> bool:
        """检查此后端是否已被覆盖注册。

        返回:
            如果后端已有覆盖注册则返回True
        """
        return self in _ATTN_OVERRIDES  # 检查当前枚举成员是否在覆盖字典中

    def clear_override(self) -> None:
        """清除此后端的覆盖注册，恢复为默认实现。"""
        _ATTN_OVERRIDES.pop(self, None)  # 从覆盖字典中移除当前后端的覆盖记录


class MambaAttentionBackendEnum(Enum, metaclass=_AttentionBackendEnumMeta):
    """Mamba注意力后端枚举类，列举所有支持的Mamba注意力后端。

    枚举值为默认的类路径字符串，可在运行时通过 register_backend() 覆盖。

    获取实际的后端类（会考虑覆盖）请使用：
        backend.get_class()
    """

    MAMBA1 = "vllm.v1.attention.backends.mamba1_attn.Mamba1AttentionBackend"  # Mamba1注意力后端
    MAMBA2 = "vllm.v1.attention.backends.mamba2_attn.Mamba2AttentionBackend"  # Mamba2注意力后端
    SHORT_CONV = "vllm.v1.attention.backends.short_conv_attn.ShortConvAttentionBackend"  # 短卷积注意力后端
    LINEAR = "vllm.v1.attention.backends.linear_attn.LinearAttentionBackend"  # 线性注意力后端
    GDN_ATTN = "vllm.v1.attention.backends.gdn_attn.GDNAttentionBackend"  # GDN注意力后端
    # 第三方/自定义后端的占位符 - 使用前必须先注册
    # 设置为None以避免与其他值为空字符串的后端产生别名冲突
    CUSTOM = None  # 自定义后端占位符

    def get_path(self, include_classname: bool = True) -> str:
        """获取此后端的类路径（会考虑覆盖注册）。

        参数:
            include_classname: 是否包含类名，默认为True

        返回:
            完全限定的类路径字符串

        异常:
            ValueError: 如果 Backend.CUSTOM 未注册就使用时抛出
        """
        path = _MAMBA_ATTN_OVERRIDES.get(self, self.value)  # 优先从Mamba覆盖字典获取路径，否则使用默认枚举值
        if not path:  # 如果路径为空（未注册的自定义后端）
            raise ValueError(  # 抛出异常提示需要先注册
                f"Backend {self.name} must be registered before use. "
                f"Use register_backend(Backend.{self.name}, 'your.module.YourClass')"
            )
        if not include_classname:  # 如果不需要包含类名
            path = path.rsplit(".", 1)[0]  # 去掉最后一个点号后面的类名部分，只保留模块路径
        return path  # 返回类路径字符串

    def get_class(self) -> "type[AttentionBackend]":
        """获取后端类对象（会考虑覆盖注册）。

        返回:
            后端类对象

        异常:
            ImportError: 如果后端类无法导入时抛出
            ValueError: 如果 Backend.CUSTOM 未注册就使用时抛出
        """
        return resolve_obj_by_qualname(self.get_path())  # 根据全限定路径解析并返回对应的类对象

    def is_overridden(self) -> bool:
        """检查此后端是否已被覆盖注册。

        返回:
            如果后端已有覆盖注册则返回True
        """
        return self in _MAMBA_ATTN_OVERRIDES  # 检查当前枚举成员是否在Mamba覆盖字典中

    def clear_override(self) -> None:
        """清除此后端的覆盖注册，恢复为默认实现。"""
        _MAMBA_ATTN_OVERRIDES.pop(self, None)  # 从Mamba覆盖字典中移除当前后端的覆盖记录


MAMBA_TYPE_TO_BACKEND_MAP = {  # Mamba类型名称到后端枚举名称的映射字典
    "mamba1": MambaAttentionBackendEnum.MAMBA1.name,  # mamba1类型映射到MAMBA1后端
    "mamba2": MambaAttentionBackendEnum.MAMBA2.name,  # mamba2类型映射到MAMBA2后端
    "short_conv": MambaAttentionBackendEnum.SHORT_CONV.name,  # 短卷积类型映射到SHORT_CONV后端
    "linear_attention": MambaAttentionBackendEnum.LINEAR.name,  # 线性注意力类型映射到LINEAR后端
    "gdn_attention": MambaAttentionBackendEnum.GDN_ATTN.name,  # GDN注意力类型映射到GDN_ATTN后端
    "custom": MambaAttentionBackendEnum.CUSTOM.name,  # 自定义类型映射到CUSTOM后端
}


_ATTN_OVERRIDES: dict[AttentionBackendEnum, str] = {}  # 注意力后端覆盖注册字典，存储被覆盖的后端及其新类路径
_MAMBA_ATTN_OVERRIDES: dict[MambaAttentionBackendEnum, str] = {}  # Mamba注意力后端覆盖注册字典


def register_backend(
    backend: AttentionBackendEnum | MambaAttentionBackendEnum,  # 要注册或覆盖的后端枚举成员
    class_path: str | None = None,  # 可选的类路径字符串，为None时作为装饰器使用
    is_mamba: bool = False,  # 是否为Mamba类型的后端，默认为False
) -> Callable[[type], type]:  # 返回装饰器函数
    """注册或覆盖一个后端实现。

    可以作为装饰器使用，也可以直接调用进行注册。

    参数:
        backend: 要注册的 AttentionBackendEnum 或 MambaAttentionBackendEnum 枚举成员
        class_path: 可选的类路径。如果未提供且作为装饰器使用，将从被装饰的类自动生成。
        is_mamba: 是否为Mamba后端，默认为False

    返回:
        如果 class_path 为 None 则返回装饰器函数，否则返回一个无操作的透传函数

    使用示例:
        # 覆盖已有的注意力后端
        @register_backend(AttentionBackendEnum.FLASH_ATTN)
        class MyCustomFlashAttn:
            ...

        # 覆盖已有的Mamba注意力后端
        @register_backend(MambaAttentionBackendEnum.LINEAR, is_mamba=True)
        class MyCustomMambaAttn:
            ...

        # 注册自定义的第三方注意力后端
        @register_backend(AttentionBackendEnum.CUSTOM)
        class MyCustomBackend:
            ...

        # 直接注册（非装饰器方式）
        register_backend(
            AttentionBackendEnum.CUSTOM,
            "my.module.MyCustomBackend"
        )
    """

    def decorator(cls: type) -> type:
        """装饰器内部函数，将被装饰的类注册到对应的覆盖字典中。"""
        if is_mamba:  # 如果是Mamba后端
            _MAMBA_ATTN_OVERRIDES[backend] = f"{cls.__module__}.{cls.__qualname__}"  # type: ignore[index]  # 将类的全限定名注册到Mamba覆盖字典
        else:  # 如果是普通注意力后端
            _ATTN_OVERRIDES[backend] = f"{cls.__module__}.{cls.__qualname__}"  # type: ignore[index]  # 将类的全限定名注册到注意力覆盖字典
        return cls  # 返回原始类（不做修改）

    if class_path is not None:  # 如果提供了类路径（直接注册方式）
        if is_mamba:  # 如果是Mamba后端
            _MAMBA_ATTN_OVERRIDES[backend] = class_path  # type: ignore[index]  # 直接将类路径注册到Mamba覆盖字典
        else:  # 如果是普通注意力后端
            _ATTN_OVERRIDES[backend] = class_path  # type: ignore[index]  # 直接将类路径注册到注意力覆盖字典
        return lambda x: x  # 返回一个透传函数（无操作装饰器）

    return decorator  # 返回装饰器函数
