# SPDX-License-Identifier: Apache-2.0  # 开源许可证标识：Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明：vLLM 项目贡献者

import time  # 导入时间模块，用于获取请求到达时间戳
from collections.abc import Mapping  # 导入 Mapping 抽象基类，用于 trace_headers 类型注解
from typing import Any, Literal  # 导入类型提示工具，Any 表示任意类型，Literal 表示字面量类型

import vllm.envs as envs  # 导入 vLLM 环境变量配置模块
from vllm.config import VllmConfig  # 导入 vLLM 总配置类
from vllm.inputs.data import (  # 从输入数据模块导入相关类型
    ProcessorInputs,  # 处理器输入类型（经过预处理后的输入）
    PromptType,  # 提示类型（用户原始输入格式）
    SingletonInputs,  # 单一输入类型（编码器或解码器的单独输入）
)  # 结束输入数据模块导入
from vllm.inputs.parse import split_enc_dec_inputs  # 导入编码器-解码器输入拆分函数
from vllm.inputs.preprocess import InputPreprocessor  # 导入输入预处理器，用于 tokenization 等预处理
from vllm.logger import init_logger  # 导入日志初始化函数
from vllm.lora.request import LoRARequest  # 导入 LoRA 请求类，用于 LoRA 适配器相关操作
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry  # 导入多模态注册表及其全局单例
from vllm.multimodal.encoder_budget import MultiModalBudget  # 导入多模态编码器预算管理类
from vllm.multimodal.inputs import (  # 从多模态输入模块导入
    MultiModalFeatureSpec,  # 多模态特征规格类，描述单个多模态输入项
)  # 结束多模态输入模块导入
from vllm.multimodal.utils import argsort_mm_positions  # 导入多模态位置排序工具函数
from vllm.platforms import current_platform  # 导入当前运行平台（如 CUDA、CPU 等）
from vllm.pooling_params import PoolingParams  # 导入池化参数类（用于 embedding/分类等任务）
from vllm.renderers import BaseRenderer, renderer_from_config  # 导入渲染器基类和从配置创建渲染器的工厂函数
from vllm.sampling_params import SamplingParams  # 导入采样参数类（用于文本生成任务）
from vllm.tasks import GENERATION_TASKS, POOLING_TASKS, SupportedTask  # 导入支持的任务类型常量和类型定义
from vllm.tokenizers import TokenizerLike  # 导入分词器协议类型
from vllm.utils import length_from_prompt_token_ids_or_embeds, random_uuid  # 导入提示长度计算函数和随机 UUID 生成函数
from vllm.utils.jsontree import json_iter_leaves  # 导入 JSON 树叶子节点迭代器
from vllm.v1.engine import EngineCoreRequest  # 导入引擎核心请求类，是最终传递给调度器的请求格式

logger = init_logger(__name__)  # 初始化当前模块的日志记录器


# [中文注释] 输入处理器：将用户提交的 prompt（文本/token IDs/多模态输入）转换为 EngineCoreRequest。
#   核心方法 process_inputs() 完成以下流程：
#     1. _validate_params — 校验 SamplingParams/PoolingParams 合法性
#     2. _validate_lora — 校验 LoRA 请求
#     3. 通过 InputPreprocessor.preprocess 进行 tokenization
#     4. 处理多模态输入：排序 mm_placeholders，构建 MultiModalFeatureSpec 列表
#     5. 校验 prompt 长度不超过 max_model_len
#     6. 构造并返回 EngineCoreRequest
#   assign_request_id() — 为请求分配内部唯一 ID（原始 ID + 8位随机后缀）
class InputProcessor:  # 输入处理器类定义
    """输入处理器类，负责将用户提交的各种格式的输入（文本、token ID、多模态数据）
    转换为引擎核心可以处理的 EngineCoreRequest 对象。

    该类是 vLLM v1 引擎的关键组件，承担了输入验证、预处理、多模态处理
    以及请求构建等核心职责。
    """

    def __init__(  # 构造函数定义
        self,  # 实例自身引用
        vllm_config: VllmConfig,  # vLLM 的全局配置对象
        renderer: BaseRenderer | None = None,  # 可选的渲染器，用于处理聊天模板等
        *,  # 以下参数必须以关键字形式传入
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,  # 多模态注册表，默认使用全局单例
    ) -> None:  # 返回值为空
        """初始化输入处理器。

        Args:
            vllm_config: vLLM 全局配置对象，包含模型配置、缓存配置等所有子配置。
            renderer: 渲染器实例，用于处理聊天模板和 tokenization。如果为 None 则从配置创建。
            mm_registry: 多模态注册表，管理所有已注册的多模态处理器。
        """
        self.vllm_config = vllm_config  # 保存 vLLM 全局配置
        self.model_config = model_config = vllm_config.model_config  # 提取并保存模型配置
        self.cache_config = vllm_config.cache_config  # 提取并保存缓存配置
        self.lora_config = vllm_config.lora_config  # 提取并保存 LoRA 配置
        self.scheduler_config = vllm_config.scheduler_config  # 提取并保存调度器配置
        self.speculative_config = vllm_config.speculative_config  # 提取并保存推测解码配置
        self.structured_outputs_config = vllm_config.structured_outputs_config  # 提取并保存结构化输出配置
        self.observability_config = vllm_config.observability_config  # 提取并保存可观测性配置

        self.generation_config_fields = model_config.try_get_generation_config()  # 尝试获取模型的生成配置字段（如 temperature 等默认值）

        self.renderer = renderer or renderer_from_config(vllm_config)  # 如果没有提供渲染器则从配置创建一个

        self.supports_mm_inputs = mm_registry.supports_multimodal_inputs(model_config)  # 检查当前模型是否支持多模态输入
        self.mm_encoder_cache_size = 0  # 初始化多模态编码器缓存大小为 0
        self.skip_prompt_length_check = False  # 初始化是否跳过提示长度检查的标志为 False
        if self.supports_mm_inputs:  # 如果模型支持多模态输入
            mm_budget = MultiModalBudget(vllm_config, mm_registry)  # 创建多模态预算管理器
            self.mm_encoder_cache_size = mm_budget.encoder_cache_size  # 获取编码器缓存大小
            self.skip_prompt_length_check = (  # 获取是否跳过提示长度检查的标志
                mm_budget.processor.info.skip_prompt_length_check
            )
            mm_budget.reset_cache()  # Not used anymore  # 重置缓存，后续不再使用该预算对象

        self.input_preprocessor = InputPreprocessor(  # 创建输入预处理器实例
            vllm_config,  # 传入全局配置
            renderer=renderer,  # 传入渲染器
            mm_registry=mm_registry,  # 传入多模态注册表
        )  # 结束 InputPreprocessor 构造

    @property  # 将方法声明为属性访问器
    def tokenizer(self) -> TokenizerLike | None:  # 分词器属性，返回分词器或 None
        """获取分词器的属性方法。

        Returns:
            分词器实例，如果渲染器没有关联分词器则返回 None。
        """
        return self.renderer.tokenizer  # 从渲染器获取分词器

    def get_tokenizer(self) -> TokenizerLike:  # 获取分词器方法，保证返回非 None
        """获取分词器（保证非 None）。

        Returns:
            分词器实例。如果不存在则抛出异常。
        """
        return self.renderer.get_tokenizer()  # 从渲染器获取分词器（内部会检查非空）

    def _validate_params(  # 参数验证方法定义
        self,  # 实例自身引用
        params: SamplingParams | PoolingParams,  # 待验证的参数，可以是采样参数或池化参数
        supported_tasks: tuple[SupportedTask, ...],  # 模型支持的任务类型元组
    ) -> None:  # 返回值为空
        """验证采样参数或池化参数是否合法。

        如果参数不合法则抛出 ValueError。

        Args:
            params: 采样参数或池化参数实例。
            supported_tasks: 当前模型支持的任务类型元组。

        Raises:
            ValueError: 当参数验证失败时抛出。
            TypeError: 当 params 既不是 SamplingParams 也不是 PoolingParams 时抛出。
        """
        """Raise `ValueError` if SamplingParams or PoolingParams is not valid."""  # 英文文档字符串：如果参数无效则抛出 ValueError
        if isinstance(params, SamplingParams):  # 如果是采样参数（生成任务）
            supported_generation_tasks = [  # 过滤出支持的生成任务列表
                task for task in supported_tasks if task in GENERATION_TASKS  # 仅保留属于生成任务类型的任务
            ]  # 结束列表推导
            if not supported_generation_tasks:  # 如果没有支持的生成任务
                raise ValueError("This model does not support generation")  # 抛出错误：该模型不支持生成

            params.verify(  # 验证采样参数的合法性
                self.model_config,  # 传入模型配置
                self.speculative_config,  # 传入推测解码配置
                self.structured_outputs_config,  # 传入结构化输出配置
                self.tokenizer,  # 传入分词器
            )  # 结束 params.verify 调用
        elif isinstance(params, PoolingParams):  # 如果是池化参数（池化任务）
            supported_pooling_tasks = [  # 过滤出支持的池化任务列表
                task for task in supported_tasks if task in POOLING_TASKS  # 仅保留属于池化任务类型的任务
            ]  # 结束列表推导
            if not supported_pooling_tasks:  # 如果没有支持的池化任务
                raise ValueError("This model does not support pooling")  # 抛出错误：该模型不支持池化

            if params.task is None:  # 如果池化参数中没有指定任务类型
                if "token_embed" in supported_pooling_tasks:  # 如果支持 token 嵌入任务
                    params.task = "token_embed"  # 默认设置为 token 嵌入任务
                elif "token_classify" in supported_pooling_tasks:  # 如果支持 token 分类任务
                    params.task = "token_classify"  # 默认设置为 token 分类任务
                elif "plugin" in supported_pooling_tasks:  # 如果支持插件任务
                    params.task = "plugin"  # 默认设置为插件任务

            if params.task not in supported_pooling_tasks:  # 如果指定的任务不在支持列表中
                raise ValueError(  # 抛出不支持的任务错误
                    f"Unsupported task: {params.task!r} "  # 显示不支持的任务名称
                    f"Supported tasks: {supported_pooling_tasks}"  # 显示支持的任务列表
                )  # 结束 ValueError 构造

            params.verify(self.model_config)  # 验证池化参数的合法性
        else:  # 如果既不是采样参数也不是池化参数
            raise TypeError(  # 抛出类型错误
                f"params must be either SamplingParams or PoolingParams, "  # 错误信息：参数必须是 SamplingParams 或 PoolingParams
                f"but got {type(params).__name__}"  # 显示实际传入的类型名称
            )  # 结束 TypeError 构造

    def _validate_lora(self, lora_request: LoRARequest | None) -> None:  # LoRA 验证方法定义
        """验证 LoRA 请求是否合法。

        Args:
            lora_request: LoRA 请求对象，为 None 时表示不使用 LoRA。

        Raises:
            ValueError: 当传入了 LoRA 请求但 LoRA 功能未启用时抛出。
        """
        if lora_request is None:  # 如果没有 LoRA 请求
            return  # 直接返回，无需验证

        # LoRA request passed in while LoRA is not enabled
        if not self.lora_config:  # 如果 LoRA 配置未启用（为 None 或 False）
            raise ValueError(  # 抛出错误
                f"Got lora_request {lora_request} but LoRA is not enabled!"  # 显示错误信息：传入了 LoRA 请求但 LoRA 未启用
            )

        if self.tokenizer is not None:  # 如果存在分词器
            logger.warning_once(  # 记录一次性警告日志
                "vLLM has deprecated support for supporting different "  # 警告：vLLM 已弃用为不同 LoRA 使用不同分词器的支持
                "tokenizers for different LoRAs. By default, vLLM uses base "  # 默认使用基础模型的分词器
                "model's tokenizer. If you are using a LoRA "  # 如果使用的 LoRA 有自己的分词器
                "with its own tokenizer, consider specifying `--tokenizer "  # 建议通过 --tokenizer 参数指定 LoRA 的分词器路径
                "[lora_path]` to use the LoRA tokenizer."  # 建议使用 LoRA 分词器路径
            )  # 结束 warning_once 调用

    def _get_mm_identifier(  # 多模态标识符获取方法定义
        self,  # 实例自身引用
        mm_hash: str,  # 多模态输入的哈希值
        lora_request: LoRARequest | None,  # LoRA 请求对象
    ) -> str:  # 返回字符串类型的标识符
        """获取多模态输入的唯一标识符。

        当启用 tower_connector_lora 时，多模态嵌入会因 LoRA 请求不同而不同，
        因此需要将 LoRA 名称作为标识符的一部分，防止缓存命中错误。

        Args:
            mm_hash: 多模态输入内容的哈希值。
            lora_request: LoRA 请求对象，可能为 None。

        Returns:
            多模态输入的唯一标识符字符串。
        """
        """
        When enable_tower_connector_lora is True, multi-modal embeddings
        vary depending on the LoRA request. Therefore, the mm_hash must be
        generated based on the LoRA request to prevent incorrect cache hits.
        """
        if (  # 条件判断：是否直接返回原始哈希值
            lora_request is None  # 如果没有 LoRA 请求
            or self.lora_config is None  # 或者 LoRA 配置为空
            or not self.lora_config.enable_tower_connector_lora  # 或者未启用 tower connector LoRA
        ):  # 以上任一条件满足时
            return mm_hash  # 直接返回原始哈希值作为标识符
        return f"{lora_request.lora_name}:{mm_hash}"  # 将 LoRA 名称与哈希值拼接作为标识符

    @staticmethod  # 声明为静态方法，不需要实例引用
    def assign_request_id(request: EngineCoreRequest):  # 分配请求 ID 的静态方法
        """为请求分配内部唯一 ID。

        将外部提供的请求 ID 保存到 external_req_id 字段，
        然后在原始 ID 后追加 8 位随机字符以确保唯一性。

        Args:
            request: 引擎核心请求对象。

        Raises:
            ValueError: 当 external_req_id 字段已被设置时抛出。
        """
        """Replace the externally supplied request ID with an internal request ID
        that adds 8 random characters in order to ensure uniqueness.
        """
        if request.external_req_id is not None:  # 如果外部请求 ID 已经被设置
            raise ValueError(  # 抛出错误
                "The external_req_id field should not be set on EngineCoreRequests"  # 错误信息：不应手动设置 external_req_id 字段
                " passed to vLLM; use the request_id field."  # 应使用 request_id 字段
            )  # 结束 ValueError 构造
        request.external_req_id = request.request_id  # 将原始请求 ID 保存为外部请求 ID
        if envs.VLLM_DISABLE_REQUEST_ID_RANDOMIZATION:  # 如果环境变量禁用了请求 ID 随机化
            logger.warning_once(  # 记录一次性警告
                "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION is set and will be "  # 警告：该环境变量已设置，将在未来版本中移除
                "removed in a future release. Duplicate externally-provided "  # 重复的外部请求 ID 可能导致故障
                "request IDs may cause failures and/or subtle correctness errors."  # 或微妙的正确性错误
            )
        else:  # 否则（正常情况下）
            request.request_id = f"{request.external_req_id}-{random_uuid():.8}"  # 在原始 ID 后追加 8 位随机 UUID 作为内部 ID

    def process_inputs(  # 输入处理核心方法定义
        self,  # 实例自身引用
        request_id: str,  # 请求的唯一标识符
        prompt: PromptType | ProcessorInputs,  # 用户输入的提示（可以是原始格式或已处理格式）
        params: SamplingParams | PoolingParams,  # 采样参数或池化参数
        supported_tasks: tuple[SupportedTask, ...],  # 模型支持的任务类型
        arrival_time: float | None = None,  # 请求到达时间戳，默认为 None
        lora_request: LoRARequest | None = None,  # LoRA 请求，默认不使用
        tokenization_kwargs: dict[str, Any] | None = None,  # 分词器额外参数
        trace_headers: Mapping[str, str] | None = None,  # 追踪头信息（用于可观测性）
        priority: int = 0,  # 请求优先级，默认为 0
        data_parallel_rank: int | None = None,  # 数据并行的 rank 编号
        resumable: bool = False,  # 是否支持可恢复请求
    ) -> EngineCoreRequest:  # 返回引擎核心请求对象
        """处理输入并构建 EngineCoreRequest。

        这是输入处理器的核心方法，负责完整的输入处理流程：
        验证参数 -> 预处理输入 -> 处理多模态数据 -> 构建请求对象。

        Args:
            request_id: 请求的唯一标识符字符串。
            prompt: 用户输入，可以是原始提示或已处理的输入。
            params: 采样参数（生成任务）或池化参数（池化任务）。
            supported_tasks: 当前模型支持的任务类型元组。
            arrival_time: 请求到达时间戳，为 None 时自动获取当前时间。
            lora_request: LoRA 适配器请求，为 None 表示不使用 LoRA。
            tokenization_kwargs: 传递给分词器的额外关键字参数。
            trace_headers: 用于请求追踪的头信息字典。
            priority: 请求优先级，数值越大优先级越高。
            data_parallel_rank: 指定数据并行的 rank，为 None 时由调度器自动分配。
            resumable: 是否为可恢复请求。

        Returns:
            构建完成的 EngineCoreRequest 对象。

        Raises:
            ValueError: 当参数验证失败或提示长度超限时抛出。
        """
        self._validate_params(params, supported_tasks)  # 验证采样/池化参数的合法性
        self._validate_lora(lora_request)  # 验证 LoRA 请求的合法性

        parallel_config = self.vllm_config.parallel_config  # 获取并行配置
        dp_size = parallel_config.data_parallel_size  # 获取数据并行大小（全局）
        dp_local_size = parallel_config.data_parallel_size_local  # 获取本地数据并行大小
        num_ranks = dp_local_size if parallel_config.local_engines_only else dp_size  # 根据是否仅使用本地引擎确定有效 rank 数量
        if data_parallel_rank is not None and not (0 <= data_parallel_rank < num_ranks):  # 如果指定了 rank 且超出有效范围
            raise ValueError(  # 抛出范围错误
                f"data_parallel_rank {data_parallel_rank} "  # 显示无效的 rank 值
                f"is out of range [0, {num_ranks})."  # 显示有效范围
            )  # 结束 ValueError 构造

        if isinstance(prompt, dict) and "type" in prompt:  # 如果输入是已处理的字典格式（包含 "type" 键）
            if tokenization_kwargs:  # 如果还传入了分词器参数
                logger.warning_once(  # 记录一次性弃用警告
                    "Passing tokenization_kwargs to InputProcessor is deprecated "  # 向 InputProcessor 传入 tokenization_kwargs 已弃用
                    "and will be removed in v0.18. You should instead pass "  # 将在 v0.18 中移除
                    "them to Renderer.render_cmpl() or Renderer.render_chat()."  # 应改为传递给渲染器方法
                )

            if arrival_time is None:  # 如果未指定到达时间
                arrival_time = prompt.get("arrival_time", time.time())  # type: ignore[assignment]  # 从输入中获取到达时间，不存在则使用当前时间

            processed_inputs: ProcessorInputs = prompt  # type: ignore[assignment]  # 直接将输入作为已处理的输入使用
        else:  # 否则（输入是原始提示格式）
            logger.warning_once(  # 记录一次性弃用警告
                "Passing raw prompts to InputProcessor is deprecated "  # 向 InputProcessor 传入原始提示已弃用
                "and will be removed in v0.18. You should instead pass "  # 将在 v0.18 中移除
                "the outputs of Renderer.render_cmpl() or Renderer.render_chat()."  # 应改为传递渲染器输出
            )

            if arrival_time is None:  # 如果未指定到达时间
                arrival_time = time.time()  # 使用当前时间作为到达时间

            processed_inputs = self.input_preprocessor.preprocess(  # 使用输入预处理器进行预处理（tokenization 等）
                prompt,  # 传入原始提示
                tokenization_kwargs=tokenization_kwargs,  # 传入分词器额外参数
            )

        current_platform.validate_request(processed_inputs, params)  # 使用当前平台验证请求（平台特定的验证逻辑）

        encoder_inputs, decoder_inputs = split_enc_dec_inputs(processed_inputs)  # 将处理后的输入拆分为编码器输入和解码器输入
        self._validate_model_inputs(encoder_inputs, decoder_inputs)  # 验证模型输入（长度检查、token ID 范围检查等）

        # Mypy can be conservative for TypedDict unions; normalize access.
        if decoder_inputs["type"] == "embeds":  # 如果解码器输入类型是嵌入向量（而非 token ID）
            prompt_token_ids = None  # token ID 设为 None
            prompt_embeds = decoder_inputs["prompt_embeds"]  # 获取提示嵌入向量
        else:  # 否则（输入类型是 token ID）
            prompt_token_ids = decoder_inputs["prompt_token_ids"]  # 获取提示的 token ID 列表
            prompt_embeds = None  # 嵌入向量设为 None

        sampling_params = None  # 初始化采样参数为 None
        pooling_params = None  # 初始化池化参数为 None
        if isinstance(params, SamplingParams):  # 如果传入的是采样参数（生成任务）
            # TODO: can we avoid cloning here in multiproc case?
            sampling_params = params.clone()  # 克隆采样参数以避免修改原始对象
            # If unset max tokens, then generate up to the max_model_len.
            if sampling_params.max_tokens is None:  # 如果未设置最大生成 token 数
                seq_len = length_from_prompt_token_ids_or_embeds(  # 计算提示的长度
                    prompt_token_ids, prompt_embeds  # 从 token ID 或嵌入向量计算
                )
                sampling_params.max_tokens = self.model_config.max_model_len - seq_len  # 设置最大生成数为模型最大长度减去提示长度

            sampling_params.update_from_generation_config(  # 从模型的生成配置更新采样参数
                self.generation_config_fields,  # 传入生成配置字段
                self.renderer.get_eos_token_id(),  # 传入结束 token ID
            )
            if self.tokenizer is not None:  # 如果分词器存在
                sampling_params.update_from_tokenizer(self.tokenizer)  # 从分词器更新采样参数（如 eos_token_id 等）
        else:  # 否则（传入的是池化参数）
            pooling_params = params.clone()  # 克隆池化参数

        # Multimodal related.
        mm_features: list[MultiModalFeatureSpec] | None = None  # 初始化多模态特征列表为 None

        if decoder_inputs["type"] == "multimodal":  # 如果解码器输入包含多模态数据
            decoder_mm_inputs = decoder_inputs["mm_kwargs"]  # 获取多模态输入数据字典
            decoder_mm_positions = decoder_inputs["mm_placeholders"]  # 获取多模态占位符位置字典
            decoder_mm_hashes = decoder_inputs["mm_hashes"]  # 获取多模态输入的哈希值字典

            if not all(  # 验证所有哈希值必须是字符串类型
                isinstance(leaf, str) for leaf in json_iter_leaves(decoder_mm_hashes)  # 遍历哈希值字典的所有叶子节点
            ):
                raise ValueError(  # 如果存在非字符串的哈希值，抛出错误
                    f"mm_hashes must contain only strings, got: {decoder_mm_hashes}. "  # 显示实际的哈希值
                    "This is likely due to an incorrect custom implementation of "  # 可能是自定义 MultiModalProcessor.apply 方法实现有误
                    "MultiModalProcessor.apply method."
                )

            # Merge and flatten multimodal placeholders, hashes and inputs
            # from dictionaries to lists, and sort them by each item's position
            # in the input sequence.
            sorted_mm_idxs = argsort_mm_positions(decoder_mm_positions)  # 按输入序列中的位置对多模态项进行排序，返回排序后的索引

            mm_features = []  # 初始化多模态特征列表
            for modality, idx in sorted_mm_idxs:  # 遍历排序后的多模态索引（模态名, 项目索引）
                base_mm_hash = decoder_mm_hashes[modality][idx]  # 获取当前多模态项的基础哈希值
                mm_features.append(  # 将多模态特征规格添加到列表
                    MultiModalFeatureSpec(  # 创建多模态特征规格对象
                        data=decoder_mm_inputs[modality][idx],  # 该多模态项的实际数据
                        modality=modality,  # 模态类型（如 "image"、"audio" 等）
                        identifier=self._get_mm_identifier(  # 获取唯一标识符（考虑 LoRA）
                            base_mm_hash,  # 传入基础哈希值
                            lora_request,  # 传入 LoRA 请求
                        ),
                        mm_position=decoder_mm_positions[modality][idx],  # 该项在输入序列中的位置信息
                        mm_hash=base_mm_hash,  # 多模态内容的哈希值
                    )
                )

        return EngineCoreRequest(  # 构建并返回引擎核心请求对象
            request_id=request_id,  # 请求唯一标识符
            prompt_token_ids=prompt_token_ids,  # 提示的 token ID 列表（可能为 None）
            prompt_embeds=prompt_embeds,  # 提示的嵌入向量（可能为 None）
            mm_features=mm_features,  # 多模态特征列表（可能为 None）
            sampling_params=sampling_params,  # 采样参数（生成任务时非 None）
            pooling_params=pooling_params,  # 池化参数（池化任务时非 None）
            arrival_time=arrival_time,  # 请求到达时间戳
            lora_request=lora_request,  # LoRA 请求对象
            cache_salt=decoder_inputs.get("cache_salt"),  # 缓存盐值，用于前缀缓存的区分
            priority=priority,  # 请求优先级
            data_parallel_rank=data_parallel_rank,  # 数据并行 rank 编号
            trace_headers=trace_headers,  # 追踪头信息
            resumable=resumable,  # 是否可恢复
        )

    def _validate_prompt_len(
        self,
        prompt_len: int,  # 提示的长度（token 数量）
        prompt_type: Literal["encoder", "decoder"],  # 提示类型：编码器或解码器
    ):
        """验证提示长度是否在允许范围内。

        Args:
            prompt_len: 提示的 token 数量。
            prompt_type: 提示类型，"encoder" 表示编码器提示，"decoder" 表示解码器提示。

        Raises:
            ValueError: 当提示为空（解码器）或超过最大模型长度时抛出。
        """
        if self.skip_prompt_length_check:  # 如果配置为跳过提示长度检查
            return  # 直接返回

        if prompt_len == 0 and prompt_type == "decoder":  # 如果解码器提示长度为 0
            raise ValueError(f"The {prompt_type} prompt cannot be empty")  # 抛出错误：解码器提示不能为空

        model_config = self.model_config  # 获取模型配置
        max_prompt_len = (  # 计算最大允许的提示长度
            model_config.max_model_len  # 解码器使用模型最大长度
            if prompt_type == "decoder"  # 如果是解码器提示
            else self.mm_encoder_cache_size  # 编码器使用多模态编码器缓存大小
        )
        if prompt_len > max_prompt_len:  # 如果提示长度超过最大限制
            if self.supports_mm_inputs:  # 如果模型支持多模态输入
                suggestion = (  # 给出多模态相关的建议信息
                    "Make sure that `max_model_len` is no smaller than the "  # 确保 max_model_len 不小于文本 token 加多模态 token 的总数
                    "number of text tokens plus multimodal tokens. For image "  # 对于图像输入
                    "inputs, the number of image tokens depends on the number "  # 图像 token 数量取决于图像数量
                    "of images, and possibly their aspect ratios as well."  # 以及可能的宽高比
                )
            else:  # 如果不支持多模态输入
                suggestion = (  # 给出纯文本相关的建议信息
                    "Make sure that `max_model_len` is no smaller than the "  # 确保 max_model_len 不小于文本 token 数量
                    "number of text tokens."
                )

            raise ValueError(  # 抛出提示过长的错误
                f"The {prompt_type} prompt (length {prompt_len}) is "  # 显示提示类型和长度
                f"longer than the maximum model length of {max_prompt_len}. "  # 显示最大模型长度
                f"{suggestion}"  # 附加建议信息
            )
        elif prompt_len == max_prompt_len and model_config.runner_type == "generate":  # 如果提示长度恰好等于最大长度且是生成任务
            suggestion = (  # 给出提示占满模型长度的建议
                "Make sure that `max_model_len` is no smaller than the "  # 确保 max_model_len 不小于提示加输出 token 的总数
                "number of text tokens (prompt + requested output tokens)."
            )
            raise ValueError(  # 抛出错误：没有空间生成输出 token
                f"The {prompt_type} prompt (length {prompt_len}) plus the number of "  # 显示提示长度
                f"requested output tokens (at least 1) is longer than the maximum "  # 加上至少 1 个输出 token 超过最大长度
                f"model length of {max_prompt_len}. {suggestion}"  # 显示最大模型长度和建议
            )

    def _validate_model_input(
        self,
        prompt_inputs: SingletonInputs,  # 单一输入数据（编码器或解码器）
        prompt_type: Literal["encoder", "decoder"],  # 输入类型标识
    ) -> None:
        """验证单个模型输入（编码器或解码器）的合法性。

        检查内容包括：提示长度、多模态项大小、token ID 范围。

        Args:
            prompt_inputs: 单一输入数据字典。
            prompt_type: "encoder" 或 "decoder"，标识输入来源。

        Raises:
            ValueError: 当输入不合法时抛出（长度超限、token ID 越界等）。
        """
        model_config = self.model_config  # 获取模型配置
        tokenizer = self.tokenizer  # 获取分词器

        prompt_ids = (  # 提取 token ID 列表
            None  # 如果是嵌入类型则为 None
            if prompt_inputs["type"] == "embeds"  # 检查输入类型是否为嵌入
            else prompt_inputs["prompt_token_ids"]  # 否则获取 token ID 列表
        )
        prompt_embeds = (  # 提取嵌入向量
            prompt_inputs["prompt_embeds"]  # 如果是嵌入类型则获取嵌入向量
            if prompt_inputs["type"] == "embeds"  # 检查输入类型是否为嵌入
            else None  # 否则为 None
        )

        prompt_len = length_from_prompt_token_ids_or_embeds(prompt_ids, prompt_embeds)  # 计算提示长度
        self._validate_prompt_len(prompt_len, prompt_type)  # 验证提示长度是否合法

        if prompt_inputs["type"] == "multimodal":  # 如果输入包含多模态数据
            decoder_mm_positions = prompt_inputs["mm_placeholders"]  # 获取多模态占位符位置
            for modality, mm_positions in decoder_mm_positions.items():  # 遍历每种模态及其位置列表
                for mm_position in mm_positions:  # 遍历该模态的每个位置项
                    embed_length = mm_position.get_num_embeds()  # 获取该多模态项的嵌入长度
                    if embed_length > self.mm_encoder_cache_size:  # 如果嵌入长度超过编码器缓存大小
                        raise ValueError(  # 抛出超出缓存大小的错误
                            f"The {prompt_type} prompt contains a(n) {modality} item "  # 显示模态类型
                            f"with length {embed_length}, which exceeds the "  # 显示嵌入长度
                            f"pre-allocated encoder cache size "  # 超过预分配的编码器缓存大小
                            f"{self.mm_encoder_cache_size}. Please reduce the input "  # 显示缓存大小
                            f"size or increase the encoder cache size "  # 建议减小输入或增大缓存
                            f"by setting --limit-mm-per-prompt at startup."  # 建议通过启动参数调整
                        )

        if prompt_ids and tokenizer is not None:  # 如果存在 token ID 且有分词器
            max_input_id = max(prompt_ids, default=0)  # 获取输入中最大的 token ID

            # NOTE: tokenizer.max_token_id is the tokenizer's vocab size while
            # self.model_config.get_vocab_size() is the model's vocab size.
            # For Qwen3 models, the language model has extra tokens that do
            # not exist in the tokenizer, and vice versa for multimodal
            # placeholder tokens in some multimodal models.
            # See https://github.com/QwenLM/Qwen3/issues/29#issuecomment-1933720399 # noqa: E501
            # and https://github.com/vllm-project/vllm/pull/22471#discussion_r2312251421 # noqa: E501

            # Here we take the max of the two to determine if a token id is
            # truly out-of-vocabulary.
            model_vocab_size = model_config.get_vocab_size()  # 获取模型的词汇表大小
            if max_input_id > max(tokenizer.max_token_id, model_vocab_size - 1):  # 如果最大 token ID 超出词汇表和模型词汇表的最大值
                raise ValueError(f"Token id {max_input_id} is out of vocabulary")  # 抛出 token ID 越界错误

    def _validate_model_inputs(
        self,
        encoder_inputs: SingletonInputs | None,  # 编码器输入，可能为 None（非编码器-解码器模型）
        decoder_inputs: SingletonInputs,  # 解码器输入（必须存在）
    ):
        """验证编码器和解码器的模型输入。

        分别对编码器输入（如果存在）和解码器输入进行验证。

        Args:
            encoder_inputs: 编码器输入数据，编码器-解码器模型时非 None。
            decoder_inputs: 解码器输入数据。
        """
        if encoder_inputs is not None:  # 如果存在编码器输入
            self._validate_model_input(encoder_inputs, prompt_type="encoder")  # 验证编码器输入

        self._validate_model_input(decoder_inputs, prompt_type="decoder")  # 验证解码器输入
