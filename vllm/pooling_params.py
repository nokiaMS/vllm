# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""池化模型参数模块。
定义用于池化模型（嵌入、分类、评分等）的API参数类。
"""

from copy import deepcopy  # 导入深拷贝函数
from typing import Any  # 导入Any类型

import msgspec  # 导入msgspec序列化库

from vllm.config import ModelConfig, PoolerConfig  # 导入模型配置和池化器配置
from vllm.sampling_params import RequestOutputKind  # 导入请求输出类型枚举
from vllm.tasks import PoolingTask  # 导入池化任务类型


class LateInteractionParams(
    msgspec.Struct,  # 继承msgspec结构体
    omit_defaults=True,  # type: ignore[call-arg]  # 序列化时省略默认值
    array_like=True,  # 使用类数组序列化格式
):  # type: ignore[call-arg]
    """Metadata for worker-side late-interaction scoring.
    工作节点侧延迟交互评分的元数据。

    Attributes:
        mode:
            - "cache_query": cache query token embeddings
            - "score_doc": score a document against a cached query.
            模式：
            - "cache_query": 缓存查询token嵌入
            - "score_doc": 对文档与缓存的查询进行评分
        query_key: stable key used for both DP routing and worker cache lookup.
            查询键：用于数据并行路由和工作节点缓存查找的稳定键。
        query_uses: expected number of document requests
            查询使用次数：预期的文档请求数量。
    """

    mode: str  # 交互模式（缓存查询或评分文档）
    query_key: str  # 查询的唯一标识键
    query_uses: int | None = None  # 预期使用次数（可选）


class PoolingParams(
    msgspec.Struct,  # 继承msgspec结构体
    omit_defaults=True,  # type: ignore[call-arg]  # 序列化时省略默认值
    array_like=True,  # 使用类数组序列化格式
):  # type: ignore[call-arg]
    """API parameters for pooling models.
    池化模型的API参数类。

    Attributes:
        use_activation: Whether to apply activation function to the pooler outputs.
            `None` uses the pooler's default, which is `True` in most cases.
            是否对池化器输出应用激活函数。`None`使用池化器默认值（通常为True）。
        dimensions: Reduce the dimensions of embeddings
            if model support matryoshka representation.
            如果模型支持套娃表示（matryoshka representation），则可降低嵌入维度。
    """

    # --8<-- [start:common-pooling-params]
    use_activation: bool | None = None  # 是否使用激活函数（None表示使用默认值）
    # --8<-- [end:common-pooling-params]

    ## for embeddings models  # 嵌入模型相关参数
    # --8<-- [start:embed-pooling-params]
    dimensions: int | None = None  # 嵌入输出维度（用于套娃表示降维）
    # --8<-- [end:embed-pooling-params]

    ## for classification, scoring and rerank  # 分类、评分和重排序相关参数
    # --8<-- [start:classify-pooling-params]
    # --8<-- [end:classify-pooling-params]

    ## for step pooling models  # 步骤池化模型相关参数
    step_tag_id: int | None = None  # 步骤标签ID
    returned_token_ids: list[int] | None = None  # 需要返回的token ID列表

    ## Internal use only  # 仅供内部使用的参数
    task: PoolingTask | None = None  # 池化任务类型
    requires_token_ids: bool = False  # 是否需要token ID
    skip_reading_prefix_cache: bool | None = None  # 是否跳过读取前缀缓存
    late_interaction_params: LateInteractionParams | None = None  # 延迟交互参数
    extra_kwargs: dict[str, Any] | None = None  # 额外的关键字参数
    output_kind: RequestOutputKind = RequestOutputKind.FINAL_ONLY  # 输出类型（默认仅最终结果）

    @property
    def all_parameters(self) -> list[str]:
        """返回所有可配置参数名称列表。"""
        return ["dimensions", "use_activation"]  # 返回所有参数名列表

    @property
    def valid_parameters(self):
        """返回每种任务类型对应的有效参数映射。"""
        return {  # 各任务类型支持的参数映射
            "embed": ["dimensions", "use_activation"],  # 嵌入任务支持的参数
            "classify": ["use_activation"],  # 分类任务支持的参数
            "score": ["use_activation"],  # 评分任务支持的参数
            "token_embed": ["dimensions", "use_activation"],  # Token嵌入任务支持的参数
            "token_classify": ["use_activation"],  # Token分类任务支持的参数
        }

    def clone(self) -> "PoolingParams":
        """Returns a deep copy of the PoolingParams instance.
        返回PoolingParams实例的深拷贝。
        """
        return deepcopy(self)  # 返回当前实例的深拷贝

    def verify(self, model_config: ModelConfig) -> None:
        """验证池化参数的有效性。
        根据模型配置验证并设置默认参数值。

        Args:
            model_config: 模型配置对象。
        """
        # plugin task uses io_processor.parse_request to verify inputs,
        # skipping PoolingParams verify
        # 插件任务使用io_processor.parse_request验证输入，跳过PoolingParams验证
        if self.task == "plugin":  # 如果是插件任务
            if self.skip_reading_prefix_cache is None:  # 如果未设置跳过前缀缓存
                self.skip_reading_prefix_cache = True  # 默认跳过前缀缓存
            return  # 直接返回，不做进一步验证

        # NOTE: Task validation needs to done against the model instance,
        # which is not available in model config. So, it's not included
        # in this method
        # 注意：任务验证需要针对模型实例进行，而模型实例在模型配置中不可用，因此不包含在此方法中
        self._merge_default_parameters(model_config)  # 合并默认参数
        self._set_default_parameters(model_config)  # 设置默认参数
        self._verify_valid_parameters()  # 验证参数有效性

    def _merge_default_parameters(self, model_config: ModelConfig) -> None:
        """合并池化器配置中的默认参数。
        将池化器配置中未被用户覆盖的参数设为默认值。

        Args:
            model_config: 模型配置对象。
        """
        pooler_config = model_config.pooler_config  # 获取池化器配置
        if pooler_config is None:  # 如果没有池化器配置
            return  # 直接返回

        assert self.task is not None, "task must be set"  # 断言任务类型已设置
        valid_parameters = self.valid_parameters[self.task]  # 获取当前任务的有效参数列表

        for k in valid_parameters:  # 遍历有效参数
            if getattr(pooler_config, k, None) is None:  # 如果池化器配置中该参数为None
                continue  # 跳过

            if getattr(self, k, None) is None:  # 如果用户未设置该参数
                setattr(self, k, getattr(pooler_config, k))  # 使用池化器配置中的默认值

        if self.skip_reading_prefix_cache is None:  # 如果未设置跳过前缀缓存选项
            # If prefix caching is enabled,
            # the output of all pooling may less than n_prompt_tokens,
            # we need to skip reading cache at this request.
            # 如果启用了前缀缓存，池化输出可能少于提示token数，需要跳过读取缓存
            if self.task in ["token_embed", "token_classify"]:  # Token级别任务
                self.skip_reading_prefix_cache = True  # 跳过前缀缓存
            else:  # 其他任务
                self.skip_reading_prefix_cache = False  # 不跳过前缀缓存

        self._verify_step_pooling(pooler_config, valid_parameters)  # 验证步骤池化参数

    def _verify_step_pooling(
        self,
        pooler_config: PoolerConfig,  # 池化器配置
        valid_parameters: list[str],  # 有效参数列表
    ):
        """验证步骤池化相关参数。
        确保步骤池化参数仅在STEP池化类型下使用。

        Args:
            pooler_config: 池化器配置。
            valid_parameters: 当前任务的有效参数列表。
        """
        step_pooling_parameters = ["step_tag_id", "returned_token_ids"]  # 步骤池化专用参数
        if pooler_config.tok_pooling_type != "STEP":  # 如果不是STEP池化类型
            invalid_parameters = []  # 无效参数列表
            for k in step_pooling_parameters:  # 遍历步骤池化参数
                if getattr(self, k, None) is not None:  # 如果用户设置了步骤池化参数
                    invalid_parameters.append(k)  # 添加到无效参数列表

            if invalid_parameters:  # 如果存在无效参数
                raise ValueError(  # 抛出值错误
                    f"Task {self.task} only supports {valid_parameters} "
                    f"parameters, does not support "
                    f"{invalid_parameters} parameters"
                )
        else:  # 如果是STEP池化类型
            for k in step_pooling_parameters:  # 遍历步骤池化参数
                if getattr(pooler_config, k, None) is None:  # 如果池化器配置中该参数为None
                    continue  # 跳过

                if getattr(self, k, None) is None:  # 如果用户未设置该参数
                    setattr(self, k, getattr(pooler_config, k))  # 使用池化器配置的默认值

    def _set_default_parameters(self, model_config: ModelConfig):
        """根据任务类型设置默认参数值。

        Args:
            model_config: 模型配置对象。

        Raises:
            ValueError: 当维度参数不合法或任务类型未知时。
        """
        if self.task in ["embed", "token_embed"]:  # 如果是嵌入类任务
            if self.use_activation is None:  # 如果未设置激活函数选项
                self.use_activation = True  # 默认启用激活函数

            if self.dimensions is not None:  # 如果指定了输出维度
                if not model_config.is_matryoshka:  # 如果模型不支持套娃表示
                    raise ValueError(  # 抛出值错误
                        f'Model "{model_config.served_model_name}" does not '
                        f"support matryoshka representation, "
                        f"changing output dimensions will lead to poor results."
                    )

                mds = model_config.matryoshka_dimensions  # 获取支持的套娃维度列表
                if mds is not None:  # 如果有指定的套娃维度
                    if self.dimensions not in mds:  # 如果请求的维度不在支持列表中
                        raise ValueError(  # 抛出值错误
                            f"Model {model_config.served_model_name!r} "
                            f"only supports {str(mds)} matryoshka dimensions, "
                            f"use other output dimensions will "
                            f"lead to poor results."
                        )
                elif self.dimensions < 1:  # 如果维度小于1
                    raise ValueError("Dimensions must be greater than 0")  # 抛出值错误

        elif self.task in ["classify", "score", "token_classify"]:  # 如果是分类或评分任务
            if self.use_activation is None:  # 如果未设置激活函数选项
                self.use_activation = True  # 默认启用激活函数
        else:  # 未知任务类型
            raise ValueError(f"Unknown pooling task: {self.task!r}")  # 抛出值错误

    def _verify_valid_parameters(self):
        """验证用户设置的参数是否对当前任务有效。

        Raises:
            ValueError: 当存在对当前任务无效的参数时。
        """
        assert self.task is not None, "task must be set"  # 断言任务类型已设置
        valid_parameters = self.valid_parameters[self.task]  # 获取当前任务的有效参数
        invalid_parameters = []  # 无效参数列表
        for k in self.all_parameters:  # 遍历所有参数
            if k in valid_parameters:  # 如果是有效参数
                continue  # 跳过

            if getattr(self, k, None) is not None:  # 如果用户设置了无效参数
                invalid_parameters.append(k)  # 添加到无效列表

        if invalid_parameters:  # 如果存在无效参数
            raise ValueError(  # 抛出值错误
                f"Task {self.task!r} only supports {valid_parameters} "
                f"parameters, does not support "
                f"{invalid_parameters} parameters"
            )

    def __repr__(self) -> str:
        """返回PoolingParams的字符串表示。"""
        return (  # 构建格式化的字符串表示
            f"PoolingParams("
            f"task={self.task}, "  # 任务类型
            f"dimensions={self.dimensions}, "  # 输出维度
            f"use_activation={self.use_activation}, "  # 是否使用激活函数
            f"step_tag_id={self.step_tag_id}, "  # 步骤标签ID
            f"returned_token_ids={self.returned_token_ids}, "  # 返回的token ID
            f"requires_token_ids={self.requires_token_ids}, "  # 是否需要token ID
            f"skip_reading_prefix_cache={self.skip_reading_prefix_cache}, "  # 是否跳过前缀缓存
            f"late_interaction_params={self.late_interaction_params}, "  # 延迟交互参数
            f"extra_kwargs={self.extra_kwargs})"  # 额外参数
        )

    def __post_init__(self) -> None:
        """初始化后的验证。
        确保输出类型为FINAL_ONLY。
        """
        assert self.output_kind == RequestOutputKind.FINAL_ONLY, (  # 断言输出类型为仅最终结果
            "For pooling output_kind has to be FINAL_ONLY"
        )
