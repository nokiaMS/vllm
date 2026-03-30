# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLABackend, AiterMLAImpl  # 从ROCm AIter MLA模块导入基类AiterMLABackend和AiterMLAImpl


class AiterTritonMLABackend(AiterMLABackend):
    """基于AIter Triton的MLA（多头潜在注意力）后端类，继承自AiterMLABackend"""

    @staticmethod  # 静态方法装饰器
    def get_name() -> str:  # 获取后端名称的方法，返回字符串
        """返回该注意力后端的名称标识"""
        return "AITER_TRITON_MLA"  # 返回后端名称字符串"AITER_TRITON_MLA"

    @staticmethod  # 静态方法装饰器
    def get_impl_cls() -> type["AiterTritonMLAImpl"]:  # 获取实现类的方法，返回AiterTritonMLAImpl类型
        """返回该后端对应的实现类"""
        return AiterTritonMLAImpl  # 返回AiterTritonMLAImpl实现类


class AiterTritonMLAImpl(AiterMLAImpl):
    """基于AIter Triton的MLA实现类，继承自AiterMLAImpl，使用Triton内核实现变长注意力计算"""

    def __init__(  # 构造函数
        self,  # 实例自身引用
        num_heads: int,  # 查询注意力头的数量
        head_size: int,  # 每个注意力头的维度大小
        scale: float,  # 注意力缩放因子
        num_kv_heads: int,  # 键值注意力头的数量
        alibi_slopes: list[float] | None,  # ALiBi位置编码的斜率列表，可为None
        sliding_window: int | None,  # 滑动窗口大小，可为None表示不使用滑动窗口
        kv_cache_dtype: str,  # 键值缓存的数据类型
        logits_soft_cap: float | None,  # 注意力logits的软上限值，可为None
        attn_type: str,  # 注意力类型
        kv_sharing_target_layer_name: str | None,  # KV共享的目标层名称，可为None
        # MLA Specific Arguments
        **mla_args,  # MLA特有的额外关键字参数
    ) -> None:  # 无返回值
        """初始化AiterTritonMLAImpl实例，调用父类构造函数并导入Triton闪存注意力函数"""
        super().__init__(  # 调用父类AiterMLAImpl的构造函数
            num_heads,  # 传递查询头数量
            head_size,  # 传递头维度大小
            scale,  # 传递缩放因子
            num_kv_heads,  # 传递键值头数量
            alibi_slopes,  # 传递ALiBi斜率
            sliding_window,  # 传递滑动窗口大小
            kv_cache_dtype,  # 传递KV缓存数据类型
            logits_soft_cap,  # 传递logits软上限
            attn_type,  # 传递注意力类型
            kv_sharing_target_layer_name,  # 传递KV共享目标层名称
            **mla_args,  # 传递MLA特有参数
        )
        from aiter.ops.triton.mha import flash_attn_varlen_func  # 从aiter的Triton多头注意力模块导入变长闪存注意力函数

        self.flash_attn_varlen_func = flash_attn_varlen_func  # 将变长闪存注意力函数保存为实例属性

    def _flash_attn_varlen_diff_headdims(  # 定义支持不同头维度的变长闪存注意力方法
        self, q, k, v, return_softmax_lse=False, softmax_scale=None, **kwargs  # 接收查询q、键k、值v，以及是否返回LSE、softmax缩放因子和其他参数
    ):
        """执行支持不同查询/键值头维度的变长闪存注意力计算，并处理LSE转置"""
        result = self.flash_attn_varlen_func(  # type: ignore[call-arg]  # 调用Triton变长闪存注意力函数
            q,  # 查询张量
            k,  # 键张量
            v,  # 值张量
            softmax_scale=softmax_scale,  # 设置softmax缩放因子
            return_lse=return_softmax_lse,  # 设置是否返回log-sum-exp值
            **kwargs,  # 传递其他关键字参数
        )
        # Transpose the LSE if Triton MHA is used:
        # (q.shape[0], num_q_heads) to (num_q_heads, q.shape[0])
        if type(result) is tuple and return_softmax_lse:  # 如果结果是元组且需要返回LSE
            output, lse = result  # 解包输出和LSE值
            lse = lse.T.contiguous()  # 对LSE进行转置并确保内存连续，从(q.shape[0], num_q_heads)变为(num_q_heads, q.shape[0])
            return (output, lse)  # 返回输出和转置后的LSE元组
        return result  # 如果不需要LSE或结果不是元组，直接返回结果
