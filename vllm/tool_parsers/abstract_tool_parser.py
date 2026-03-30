# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib  # 导入动态模块导入工具
import os  # 导入操作系统接口模块
from collections.abc import Callable, Sequence  # 导入可调用对象和序列的抽象基类
from functools import cached_property  # 导入缓存属性装饰器

from openai.types.responses.response_format_text_json_schema_config import (  # 从OpenAI类型导入
    ResponseFormatTextJSONSchemaConfig,  # 导入JSON Schema响应格式配置
)

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest  # 导入聊天补全请求协议
from vllm.entrypoints.openai.engine.protocol import (  # 从引擎协议模块导入
    DeltaMessage,  # 导入增量消息类
    ExtractedToolCallInformation,  # 导入提取的工具调用信息类
)
from vllm.entrypoints.openai.responses.protocol import (  # 从响应协议模块导入
    ResponsesRequest,  # 导入响应请求类
    ResponseTextConfig,  # 导入响应文本配置类
)
from vllm.logger import init_logger  # 导入日志初始化工具
from vllm.sampling_params import (  # 从采样参数模块导入
    StructuredOutputsParams,  # 导入结构化输出参数类
)
from vllm.tokenizers import TokenizerLike  # 导入分词器类型接口
from vllm.tool_parsers.utils import get_json_schema_from_tools  # 导入从工具获取JSON Schema的工具函数
from vllm.utils.collection_utils import is_list_of  # 导入列表类型检查工具
from vllm.utils.import_utils import import_from_path  # 导入从路径导入模块的工具

logger = init_logger(__name__)  # 初始化当前模块的日志记录器


class ToolParser:
    """
    Abstract ToolParser class that should not be used directly. Provided
    properties and methods should be used in
    derived classes.
    抽象工具解析器类，不应直接使用。提供的属性和方法应在派生类中使用。
    """

    def __init__(self, tokenizer: TokenizerLike):  # 初始化工具解析器
        """
        初始化工具解析器。
        Args:
            tokenizer: 模型分词器实例
        """
        self.prev_tool_call_arr: list[dict] = []  # 之前的工具调用数组
        # the index of the tool call that is currently being parsed
        self.current_tool_id: int = -1  # 当前正在解析的工具调用索引
        self.current_tool_name_sent: bool = False  # 当前工具名称是否已发送
        self.streamed_args_for_tool: list[str] = []  # 每个工具已流式传输的参数列表

        self.model_tokenizer = tokenizer  # 存储模型分词器

    @cached_property  # 缓存属性，只计算一次
    def vocab(self) -> dict[str, int]:  # 获取词汇表
        """
        获取分词器的词汇表。
        """
        # NOTE: Only PreTrainedTokenizerFast is guaranteed to have .vocab
        # whereas all tokenizers have .get_vocab()
        return self.model_tokenizer.get_vocab()  # 返回词汇表字典

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:  # 调整请求参数
        """
        Static method that used to adjust the request parameters.
        用于调整请求参数的静态方法。
        """
        if not request.tools:  # 如果请求中没有工具定义
            return request  # 直接返回原始请求
        json_schema_from_tool = get_json_schema_from_tools(  # 从工具列表中获取JSON Schema
            tool_choice=request.tool_choice, tools=request.tools
        )
        # Set structured output params for tool calling
        if json_schema_from_tool is not None:  # 如果成功获取到JSON Schema
            if isinstance(request, ChatCompletionRequest):  # 如果是聊天补全请求
                # tool_choice: "Forced Function" or "required" will override
                # structured output json settings to make tool calling work correctly
                request.structured_outputs = StructuredOutputsParams(  # 设置结构化输出参数
                    json=json_schema_from_tool  # type: ignore[call-arg]
                )
                request.response_format = None  # 清除响应格式设置
            if isinstance(request, ResponsesRequest):  # 如果是响应API请求
                request.text = ResponseTextConfig()  # 创建响应文本配置
                request.text.format = ResponseFormatTextJSONSchemaConfig(  # 设置JSON Schema格式
                    name="tool_calling_response",  # 格式名称
                    schema=json_schema_from_tool,  # JSON Schema定义
                    type="json_schema",  # 类型为json_schema
                    description="Response format for tool calling",  # 描述
                    strict=True,  # 严格模式
                )

        return request  # 返回调整后的请求

    def extract_tool_calls(  # 从完整模型输出中提取工具调用
        self, model_output: str, request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        """
        Static method that should be implemented for extracting tool calls from
        a complete model-generated string.
        Used for non-streaming responses where we have the entire model response
        available before sending to the client.
        Static because it's stateless.
        应实现的静态方法，用于从完整的模型生成字符串中提取工具调用。
        用于非流式响应，其中在发送给客户端之前我们拥有完整的模型响应。
        """
        raise NotImplementedError(  # 抛出未实现错误
            "AbstractToolParser.extract_tool_calls has not been implemented!"
        )

    def extract_tool_calls_streaming(  # 从流式输出中提取工具调用
        self,
        previous_text: str,  # 之前的文本
        current_text: str,  # 当前的文本
        delta_text: str,  # 增量文本
        previous_token_ids: Sequence[int],  # 之前的token ID序列
        current_token_ids: Sequence[int],  # 当前的token ID序列
        delta_token_ids: Sequence[int],  # 增量token ID序列
        request: ChatCompletionRequest,  # 聊天补全请求
    ) -> DeltaMessage | None:
        """
        Instance method that should be implemented for extracting tool calls
        from an incomplete response; for use when handling tool calls and
        streaming. Has to be an instance method because  it requires state -
        the current tokens/diffs, but also the information about what has
        previously been parsed and extracted (see constructor)
        应实现的实例方法，用于从不完整的响应中提取工具调用；
        用于处理工具调用和流式传输。必须是实例方法，因为它需要状态 -
        当前的token/差异，以及之前已解析和提取的信息（参见构造函数）。
        """
        raise NotImplementedError(  # 抛出未实现错误
            "AbstractToolParser.extract_tool_calls_streaming has not been implemented!"
        )


class ToolParserManager:
    """
    Central registry for ToolParser implementations.

    Supports two modes:
      - Eager (immediate) registration via `register_module`
      - Lazy registration via `register_lazy_module`

    ToolParser实现的中央注册表。
    支持两种模式：
      - 通过 `register_module` 立即注册
      - 通过 `register_lazy_module` 延迟注册
    """

    tool_parsers: dict[str, type[ToolParser]] = {}  # 已注册的工具解析器字典
    lazy_parsers: dict[str, tuple[str, str]] = {}  # 延迟加载的解析器字典，名称 -> (模块路径, 类名)

    @classmethod  # 类方法
    def get_tool_parser(cls, name: str) -> type[ToolParser]:  # 获取工具解析器类
        """
        Retrieve a registered or lazily registered ToolParser class.

        If the parser is lazily registered,
        it will be imported and cached on first access.
        Raises KeyError if not found.

        检索已注册或延迟注册的ToolParser类。
        如果解析器是延迟注册的，它将在首次访问时导入并缓存。
        如果未找到则引发KeyError。
        """
        if name in cls.tool_parsers:  # 如果在已注册的解析器中找到
            return cls.tool_parsers[name]  # 直接返回

        if name in cls.lazy_parsers:  # 如果在延迟加载的解析器中找到
            return cls._load_lazy_parser(name)  # 加载并返回

        raise KeyError(f"Tool parser '{name}' not found.")  # 未找到，抛出KeyError

    @classmethod  # 类方法
    def _load_lazy_parser(cls, name: str) -> type[ToolParser]:  # 加载延迟注册的解析器
        """Import and register a lazily loaded parser.
        导入并注册一个延迟加载的解析器。"""
        module_path, class_name = cls.lazy_parsers[name]  # 获取模块路径和类名
        try:
            mod = importlib.import_module(module_path)  # 动态导入模块
            parser_cls = getattr(mod, class_name)  # 获取解析器类
            if not issubclass(parser_cls, ToolParser):  # 检查是否是ToolParser的子类
                raise TypeError(  # 抛出类型错误
                    f"{class_name} in {module_path} is not a ToolParser subclass."
                )
            cls.tool_parsers[name] = parser_cls  # 缓存解析器类
            return parser_cls  # 返回解析器类
        except Exception as e:  # 捕获异常
            logger.exception(  # 记录异常日志
                "Failed to import lazy tool parser '%s' from %s: %s",
                name,
                module_path,
                e,
            )
            raise  # 重新抛出异常

    @classmethod  # 类方法
    def _register_module(  # 立即注册工具解析器类
        cls,
        module: type[ToolParser],  # 要注册的解析器类
        module_name: str | list[str] | None = None,  # 注册名称
        force: bool = True,  # 是否强制覆盖已有注册
    ) -> None:
        """Register a ToolParser class immediately.
        立即注册一个ToolParser类。"""
        if not issubclass(module, ToolParser):  # 检查是否是ToolParser的子类
            raise TypeError(  # 抛出类型错误
                f"module must be subclass of ToolParser, but got {type(module)}"
            )

        if module_name is None:  # 如果未指定名称
            module_name = module.__name__  # 使用类名作为注册名称

        if isinstance(module_name, str):  # 如果名称是字符串
            module_names = [module_name]  # 转为列表
        elif is_list_of(module_name, str):  # 如果名称是字符串列表
            module_names = module_name  # 直接使用
        else:
            raise TypeError("module_name must be str, list[str], or None.")  # 抛出类型错误

        for name in module_names:  # 遍历所有注册名称
            if not force and name in cls.tool_parsers:  # 如果不强制且已存在
                existed = cls.tool_parsers[name]  # 获取已存在的解析器
                raise KeyError(f"{name} is already registered at {existed.__module__}")  # 抛出键错误
            cls.tool_parsers[name] = module  # 注册解析器

    @classmethod  # 类方法
    def register_lazy_module(cls, name: str, module_path: str, class_name: str) -> None:  # 注册延迟加载模块
        """
        Register a lazy module mapping.
        注册延迟加载的模块映射。

        Example:
            ToolParserManager.register_lazy_module(
                name="kimi_k2",
                module_path="vllm.tool_parsers.kimi_k2_parser",
                class_name="KimiK2ToolParser",
            )
        """
        cls.lazy_parsers[name] = (module_path, class_name)  # 存储模块路径和类名的映射

    @classmethod  # 类方法
    def register_module(  # 注册模块（支持装饰器和直接调用两种方式）
        cls,
        name: str | list[str] | None = None,  # 注册名称
        force: bool = True,  # 是否强制覆盖
        module: type[ToolParser] | None = None,  # 要注册的模块
    ) -> type[ToolParser] | Callable[[type[ToolParser]], type[ToolParser]]:
        """
        Register module immediately or lazily (as a decorator).
        立即注册模块或延迟注册（作为装饰器使用）。

        Usage:
            @ToolParserManager.register_module("kimi_k2")
            class KimiK2ToolParser(ToolParser):
                ...

        Or:
            ToolParserManager.register_module(module=SomeToolParser)
        """
        if not isinstance(force, bool):  # 检查force参数类型
            raise TypeError(f"force must be a boolean, but got {type(force)}")

        # Immediate registration
        if module is not None:  # 如果提供了模块，立即注册
            cls._register_module(module=module, module_name=name, force=force)
            return module  # 返回已注册的模块

        # Decorator usage
        def _decorator(obj: type[ToolParser]) -> type[ToolParser]:  # 装饰器函数
            module_path = obj.__module__  # 获取模块路径
            class_name = obj.__name__  # 获取类名

            if isinstance(name, str):  # 处理名称参数
                names = [name]
            elif name is not None and is_list_of(name, str):
                names = name
            else:
                names = [class_name]

            for n in names:  # 遍历所有名称
                # Lazy mapping only: do not import now
                cls.lazy_parsers[n] = (module_path, class_name)  # 注册延迟映射

            return obj  # 返回原始类

        return _decorator  # 返回装饰器

    @classmethod  # 类方法
    def list_registered(cls) -> list[str]:  # 列出所有已注册的解析器名称
        """Return names of all eagerly and lazily registered tool parsers.
        返回所有已注册和延迟注册的工具解析器名称。"""
        return sorted(set(cls.tool_parsers.keys()) | set(cls.lazy_parsers.keys()))  # 合并并排序

    @classmethod  # 类方法
    def import_tool_parser(cls, plugin_path: str) -> None:  # 从文件路径导入用户定义的解析器
        """Import a user-defined parser file from arbitrary path.
        从任意路径导入用户定义的解析器文件。"""

        module_name = os.path.splitext(os.path.basename(plugin_path))[0]  # 提取模块名称
        try:
            import_from_path(module_name, plugin_path)  # 从路径导入模块
        except Exception:  # 捕获异常
            logger.exception(  # 记录异常日志
                "Failed to load module '%s' from %s.", module_name, plugin_path
            )
