# vLLM CLI 命令参数完整手册

> 基于源码分析，涵盖所有 `vllm` 子命令及其参数。
> 入口文件：`vllm/entrypoints/cli/main.py`

---

## 目录

- [1. 全局参数](#1-全局参数)
- [2. vllm serve - 启动推理服务](#2-vllm-serve---启动推理服务)
  - [2.1 Serve 专属参数](#21-serve-专属参数)
  - [2.2 Frontend 前端服务参数](#22-frontend-前端服务参数)
  - [2.3 Engine 引擎参数](#23-engine-引擎参数)
- [3. vllm chat - 交互式对话](#3-vllm-chat---交互式对话)
- [4. vllm complete - 文本补全](#4-vllm-complete---文本补全)
- [5. vllm run-batch - 批量推理](#5-vllm-run-batch---批量推理)
- [6. vllm launch render - 无GPU渲染服务](#6-vllm-launch-render---无gpu渲染服务)
- [7. vllm bench - 性能基准测试](#7-vllm-bench---性能基准测试)
- [8. vllm collect-env - 环境信息收集](#8-vllm-collect-env---环境信息收集)

---

## 1. 全局参数

| 参数 | 说明 | 示例 | 示例功能 |
|------|------|------|----------|
| `-v` / `--version` | 显示 vLLM 版本号并退出 | `vllm --version` | 查看当前安装的 vLLM 版本 |

---

## 2. vllm serve - 启动推理服务

**作用**：启动本地 OpenAI 兼容的 API 服务器，通过 HTTP 提供 LLM 推理服务。

**基本用法**：`vllm serve [model_tag] [options]`

### 2.1 Serve 专属参数

| 参数 | 类型 | 默认值 | 作用 | 参数值说明 | 示例 | 示例功能 |
|------|------|--------|------|-----------|------|----------|
| `model_tag` | str | 无 (位置参数) | 指定要加载的模型路径或 HuggingFace 模型 ID | 本地路径或 HF 模型名称 | `vllm serve Qwen/Qwen2-7B` | 启动 Qwen2-7B 模型的推理服务 |
| `--headless` | bool | False | 无头模式运行，不启动 API 服务器，仅启动引擎进程 | 开关参数，无需赋值 | `vllm serve meta-llama/Llama-3-8B --headless --data-parallel-size 2` | 在多节点数据并行中作为 worker 节点运行 |
| `--api-server-count` / `-asc` | int | None | 指定启动的 API 服务器进程数 | 正整数；默认等于 data_parallel_size | `vllm serve model --api-server-count 4` | 启动 4 个 API 服务器进程处理并发请求 |
| `--config` | str | None | 从 YAML 配置文件读取所有 CLI 参数 | YAML 文件路径 | `vllm serve --config serve_config.yaml` | 从配置文件加载所有参数，避免命令行过长 |
| `--grpc` | bool | False | 启动 gRPC 服务器替代 HTTP 服务器 | 开关参数；需安装 `pip install vllm[grpc]` | `vllm serve model --grpc` | 使用 gRPC 协议提供服务（更低延迟） |

### 2.2 Frontend 前端服务参数

#### 2.2.1 网络与服务器配置

| 参数 | 类型 | 默认值 | 作用 | 参数值说明 | 示例 | 示例功能 |
|------|------|--------|------|-----------|------|----------|
| `--host` | str | None | 服务器监听的主机地址 | IP 地址或主机名；None 表示默认 0.0.0.0 | `vllm serve model --host 0.0.0.0` | 监听所有网络接口，允许外部访问 |
| `--port` | int | 8000 | 服务器监听的端口号 | 有效端口号 (1-65535) | `vllm serve model --port 8080` | 在 8080 端口启动服务 |
| `--uds` | str | None | Unix 域套接字路径，设置后忽略 host 和 port | Unix socket 文件路径 | `vllm serve model --uds /tmp/vllm.sock` | 通过 Unix 套接字通信（同机通信更高效） |
| `--root-path` | str | None | FastAPI 的 root_path，用于反向代理场景 | URL 前缀路径 | `vllm serve model --root-path /api/v1` | 在反向代理后以 /api/v1 为前缀提供服务 |

#### 2.2.2 SSL/TLS 加密

| 参数 | 类型 | 默认值 | 作用 | 参数值说明 | 示例 | 示例功能 |
|------|------|--------|------|-----------|------|----------|
| `--ssl-keyfile` | str | None | SSL 私钥文件路径 | PEM 格式密钥文件路径 | `vllm serve model --ssl-keyfile key.pem --ssl-certfile cert.pem` | 启用 HTTPS 加密传输 |
| `--ssl-certfile` | str | None | SSL 证书文件路径 | PEM 格式证书文件路径 | (同上) | (同上) |
| `--ssl-ca-certs` | str | None | CA 证书文件路径 | CA 证书路径 | `vllm serve model --ssl-ca-certs ca.pem` | 配置 CA 证书链 |
| `--enable-ssl-refresh` | bool | False | SSL 证书文件变更时自动刷新 | 开关参数 | `vllm serve model --enable-ssl-refresh --ssl-keyfile key.pem --ssl-certfile cert.pem` | 证书续期后自动加载新证书 |
| `--ssl-cert-reqs` | int | 0 | 是否要求客户端证书 | 0=不需要，1=可选，2=必须 | `vllm serve model --ssl-cert-reqs 2` | 要求客户端必须提供证书（双向 TLS） |
| `--ssl-ciphers` | str | None | SSL 密码套件（TLS 1.2 及以下） | 密码套件字符串 | `vllm serve model --ssl-ciphers 'ECDHE-RSA-AES256-GCM-SHA384'` | 限制使用特定的加密算法 |

#### 2.2.3 CORS 跨域配置

| 参数 | 类型 | 默认值 | 作用 | 参数值说明 | 示例 | 示例功能 |
|------|------|--------|------|-----------|------|----------|
| `--allow-credentials` | bool | False | 是否允许跨域请求携带凭据 | 开关参数 | `vllm serve model --allow-credentials` | 允许浏览器跨域请求携带 Cookie |
| `--allowed-origins` | JSON | `["*"]` | 允许的跨域来源列表 | JSON 数组格式 | `vllm serve model --allowed-origins '["https://app.com"]'` | 仅允许 app.com 域名的跨域请求 |
| `--allowed-methods` | JSON | `["*"]` | 允许的 HTTP 方法列表 | JSON 数组格式 | `vllm serve model --allowed-methods '["GET","POST"]'` | 仅允许 GET 和 POST 跨域请求 |
| `--allowed-headers` | JSON | `["*"]` | 允许的请求头列表 | JSON 数组格式 | `vllm serve model --allowed-headers '["Authorization"]'` | 仅允许携带 Authorization 头 |

#### 2.2.4 认证与安全

| 参数 | 类型 | 默认值 | 作用 | 参数值说明 | 示例 | 示例功能 |
|------|------|--------|------|-----------|------|----------|
| `--api-key` | list[str] | None | API 密钥，请求需在 header 中提供 | 一个或多个密钥字符串 | `vllm serve model --api-key "sk-mykey123"` | 启用 API 认证，只有提供正确 key 才能访问 |
| `--disable-fastapi-docs` | bool | False | 禁用 FastAPI 的 Swagger/ReDoc 文档页面 | 开关参数 | `vllm serve model --disable-fastapi-docs` | 生产环境隐藏 API 文档页面 |
| `--enable-offline-docs` | bool | False | 启用离线 API 文档（内网环境） | 开关参数 | `vllm serve model --enable-offline-docs` | 在无外网的环境中使用本地静态资源显示文档 |

#### 2.2.5 日志与监控

| 参数 | 类型 | 默认值 | 作用 | 参数值说明 | 示例 | 示例功能 |
|------|------|--------|------|-----------|------|----------|
| `--uvicorn-log-level` | str | "info" | Uvicorn 服务器日志级别 | critical/error/warning/info/debug/trace | `vllm serve model --uvicorn-log-level debug` | 开启 Uvicorn 详细调试日志 |
| `--disable-uvicorn-access-log` | bool | False | 禁用 Uvicorn 的访问日志 | 开关参数 | `vllm serve model --disable-uvicorn-access-log` | 减少高频请求产生的日志量 |
| `--disable-access-log-for-endpoints` | str | None | 对特定端点禁用访问日志 | 逗号分隔的路径列表 | `vllm serve model --disable-access-log-for-endpoints "/health,/metrics"` | 避免健康检查和指标端点产生大量日志 |
| `--log-config-file` | str | None | 日志配置 JSON 文件路径 | JSON 配置文件路径 | `vllm serve model --log-config-file log_config.json` | 自定义日志格式和输出目标 |
| `--max-log-len` | int | None | 日志中显示的最大 prompt 字符数 | 正整数；None=不限制 | `vllm serve model --max-log-len 100` | 限制日志中 prompt 最多显示 100 个字符 |
| `--enable-log-outputs` | bool | False | 在日志中记录模型输出内容 | 开关参数；需同时启用 --enable-log-requests | `vllm serve model --enable-log-requests --enable-log-outputs` | 调试时查看模型生成的完整输出内容 |
| `--enable-log-deltas` | bool | True | 是否记录输出增量 | 开关参数；仅在 --enable-log-outputs 时有效 | `vllm serve model --enable-log-outputs --no-enable-log-deltas` | 禁用流式输出的增量日志 |
| `--log-error-stack` | bool | False | 记录错误响应的堆栈信息 | 开关参数 | `vllm serve model --log-error-stack` | 调试时查看完整错误堆栈 |
| `--enable-log-requests` | bool | False | 启用请求信息日志记录 | 开关参数；INFO 级别记录请求 ID，DEBUG 级别记录 prompt | `vllm serve model --enable-log-requests` | 跟踪所有请求的 ID 和参数 |
| `--disable-log-stats` | bool | False | 禁用引擎统计日志 | 开关参数 | `vllm serve model --disable-log-stats` | 减少日志输出，提高性能 |
| `--aggregate-engine-logging` | bool | False | 数据并行时记录聚合统计而非逐引擎统计 | 开关参数 | `vllm serve model --aggregate-engine-logging -dp 4` | 多引擎时合并统计日志，减少日志量 |

#### 2.2.6 中间件与扩展

| 参数 | 类型 | 默认值 | 作用 | 参数值说明 | 示例 | 示例功能 |
|------|------|--------|------|-----------|------|----------|
| `--middleware` | list[str] | [] | 添加自定义 ASGI 中间件 | Python 导入路径；可多次指定 | `vllm serve model --middleware my_pkg.auth_middleware` | 添加自定义认证中间件 |
| `--enable-request-id-headers` | bool | False | 在响应中添加 X-Request-Id 头 | 开关参数 | `vllm serve model --enable-request-id-headers` | 为每个响应添加唯一请求 ID，便于追踪 |

#### 2.2.7 HTTP 协议配置

| 参数 | 类型 | 默认值 | 作用 | 参数值说明 | 示例 | 示例功能 |
|------|------|--------|------|-----------|------|----------|
| `--h11-max-incomplete-event-size` | int | 4194304 | h11 解析器的最大不完整事件大小(字节) | 正整数 (字节数) | `vllm serve model --h11-max-incomplete-event-size 8388608` | 增大到 8MB 以处理超大请求头 |
| `--h11-max-header-count` | int | 256 | 允许的最大 HTTP 请求头数量 | 正整数 | `vllm serve model --h11-max-header-count 512` | 允许更多请求头（复杂代理场景） |

#### 2.2.8 Chat & Tool 相关

| 参数 | 类型 | 默认值 | 作用 | 参数值说明 | 示例 | 示例功能 |
|------|------|--------|------|-----------|------|----------|
| `--chat-template` | str | None | 自定义聊天模板文件路径或单行模板 | Jinja2 模板文件路径或模板字符串 | `vllm serve model --chat-template template.jinja` | 使用自定义聊天模板格式化对话 |
| `--chat-template-content-format` | str | "auto" | 聊天模板中消息内容的渲染格式 | "auto"/"string"/"openai" | `vllm serve model --chat-template-content-format openai` | 使用 OpenAI 格式渲染消息内容（支持多模态） |
| `--trust-request-chat-template` | bool | False | 是否信任请求中携带的聊天模板 | 开关参数 | `vllm serve model --trust-request-chat-template` | 允许客户端在请求中指定自定义模板 |
| `--default-chat-template-kwargs` | JSON | None | 聊天模板默认关键字参数 | JSON 字典格式 | `vllm serve model --default-chat-template-kwargs '{"enable_thinking": false}'` | 默认禁用 Qwen3 的思考模式 |
| `--response-role` | str | "assistant" | 生成时使用的角色名称 | 角色名字符串 | `vllm serve model --response-role "bot"` | 将响应角色从 assistant 改为 bot |
| `--enable-auto-tool-choice` | bool | False | 启用自动工具调用选择 | 开关参数；需配合 --tool-call-parser | `vllm serve model --enable-auto-tool-choice --tool-call-parser hermes` | 允许模型自动选择并调用工具 |
| `--tool-call-parser` | str | None | 工具调用解析器类型 | 内置解析器名或插件注册名 | `vllm serve model --enable-auto-tool-choice --tool-call-parser llama3_json` | 使用 Llama3 JSON 格式解析工具调用 |
| `--tool-parser-plugin` | str | "" | 自定义工具解析器插件路径 | Python 模块导入路径 | `vllm serve model --tool-parser-plugin my_plugin --tool-call-parser my_parser` | 加载自定义工具解析插件 |
| `--tool-server` | str | None | 工具服务器地址 | 逗号分隔的 host:port 或 "demo" | `vllm serve model --tool-server "127.0.0.1:9000"` | 连接外部工具服务器提供函数调用能力 |
| `--exclude-tools-when-tool-choice-none` | bool | False | tool_choice='none' 时排除工具定义 | 开关参数 | `vllm serve model --exclude-tools-when-tool-choice-none` | 当不使用工具时从 prompt 中移除工具定义以节省 token |

#### 2.2.9 LoRA 相关

| 参数 | 类型 | 默认值 | 作用 | 参数值说明 | 示例 | 示例功能 |
|------|------|--------|------|-----------|------|----------|
| `--lora-modules` | list | None | LoRA 模块配置 | `name=path` 格式或 JSON 格式 | `vllm serve model --enable-lora --lora-modules my_lora=/path/to/lora` | 加载名为 my_lora 的 LoRA 适配器 |

#### 2.2.10 其他前端参数

| 参数 | 类型 | 默认值 | 作用 | 参数值说明 | 示例 | 示例功能 |
|------|------|--------|------|-----------|------|----------|
| `--return-tokens-as-token-ids` | bool | False | 将 token 表示为 `token_id:{id}` 格式 | 开关参数 | `vllm serve model --return-tokens-as-token-ids --max-logprobs 10` | 确保不可 JSON 编码的 token 能正确返回 |
| `--disable-frontend-multiprocessing` | bool | False | 在同一进程中运行前端和引擎 | 开关参数 | `vllm serve model --disable-frontend-multiprocessing` | 简化调试，避免多进程问题 |
| `--enable-prompt-tokens-details` | bool | False | 在 usage 中启用 prompt_tokens_details | 开关参数 | `vllm serve model --enable-prompt-tokens-details` | 获取 prompt token 的详细统计信息 |
| `--enable-server-load-tracking` | bool | False | 启用服务器负载指标跟踪 | 开关参数 | `vllm serve model --enable-server-load-tracking` | 在应用状态中记录服务器负载指标 |
| `--enable-force-include-usage` | bool | False | 每个请求强制包含 usage 信息 | 开关参数 | `vllm serve model --enable-force-include-usage` | 即使流式响应也在每条消息中包含 token 用量 |
| `--enable-tokenizer-info-endpoint` | bool | False | 启用 /tokenizer_info 端点 | 开关参数 | `vllm serve model --enable-tokenizer-info-endpoint` | 暴露 tokenizer 配置信息 API |
| `--tokens-only` | bool | False | 仅启用 Tokens In/Out 端点 | 开关参数；用于分离式部署 | `vllm serve model --tokens-only` | 在 Disaggregated Everything 架构中使用 |

### 2.3 Engine 引擎参数

#### 2.3.1 模型配置 (ModelConfig)

| 参数 | 类型 | 默认值 | 作用 | 参数值说明 | 示例 | 示例功能 |
|------|------|--------|------|-----------|------|----------|
| `--model` | str | "Qwen/Qwen3-0.6B" | 模型名称或路径 | HuggingFace ID 或本地路径 | `vllm serve --model meta-llama/Llama-3-8B` | 加载 Llama-3-8B 模型 |
| `--model-weights` | str | None | 单独指定模型权重路径 | 权重文件目录路径 | `vllm serve model --model-weights /data/weights` | 从不同于配置的路径加载权重 |
| `--tokenizer` | str | None | 分词器路径（默认与模型相同） | HuggingFace ID 或本地路径 | `vllm serve model --tokenizer /path/to/tokenizer` | 使用与模型不同的分词器 |
| `--hf-config-path` | str | None | HuggingFace 配置文件路径 | 配置文件目录路径 | `vllm serve model --hf-config-path /path/to/config` | 使用自定义模型配置 |
| `--served-model-name` | str | None | API 中显示的模型名称 | 字符串或逗号分隔的多个名称 | `vllm serve model --served-model-name "gpt-4-turbo"` | 在 /v1/models API 中显示自定义名称 |
| `--dtype` | str | "auto" | 模型权重和计算的数据类型 | auto/half/float16/bfloat16/float/float32 | `vllm serve model --dtype bfloat16` | 使用 bfloat16 精度运行模型 |
| `--tokenizer-mode` | str | "auto" | 分词器模式 | auto/slow/mistral/custom | `vllm serve model --tokenizer-mode slow` | 使用慢速 Python 分词器（兼容性更好） |
| `--trust-remote-code` | bool | False | 信任远程代码（HuggingFace 自定义模型） | 开关参数 | `vllm serve model --trust-remote-code` | 允许执行模型仓库中的自定义 Python 代码 |
| `--revision` | str | None | 模型版本/分支 | Git commit hash 或分支名 | `vllm serve model --revision main` | 加载模型仓库的 main 分支 |
| `--code-revision` | str | None | 代码版本（用于自定义代码模型） | Git commit hash 或分支名 | `vllm serve model --code-revision abc123` | 指定自定义代码的特定版本 |
| `--tokenizer-revision` | str | None | 分词器版本 | Git commit hash 或分支名 | `vllm serve model --tokenizer-revision v1.0` | 使用特定版本的分词器 |
| `--hf-token` | str/bool | None | HuggingFace API Token | Token 字符串或 True（使用缓存token） | `vllm serve model --hf-token "hf_xxx"` | 认证下载私有模型 |
| `--hf-overrides` | JSON | None | 覆盖 HuggingFace 配置 | JSON 格式的配置覆盖 | `vllm serve model --hf-overrides '{"num_hidden_layers": 2}'` | 修改模型架构参数（如减少层数用于测试） |
| `--max-model-len` | int | None | 最大上下文长度 | 正整数 (token 数) | `vllm serve model --max-model-len 4096` | 限制最大上下文为 4096 tokens（节省显存） |
| `--max-logprobs` | int | 20 | 最大返回的 logprobs 数 | 非负整数 | `vllm serve model --max-logprobs 50` | 允许请求最多 50 个 logprobs |
| `--seed` | int | 0 | 随机种子 | 整数 | `vllm serve model --seed 42` | 设置固定随机种子保证可复现性 |
| `--quantization` / `-q` | str | None | 量化方法 | awq/gptq/squeezellm/fp8/bitsandbytes 等 | `vllm serve model -q awq` | 使用 AWQ 量化加载模型（减少显存） |
| `--enforce-eager` | bool | False | 强制使用 eager 模式（禁用 CUDA Graph） | 开关参数 | `vllm serve model --enforce-eager` | 调试时禁用 CUDA Graph 优化 |
| `--config-format` | str | "auto" | 模型配置格式 | auto/hf/mistral | `vllm serve model --config-format hf` | 指定使用 HuggingFace 格式的配置 |
| `--disable-sliding-window` | bool | False | 禁用滑动窗口注意力 | 开关参数 | `vllm serve model --disable-sliding-window` | 强制使用完整注意力（某些模型兼容性） |
| `--disable-cascade-attn` | bool | False | 禁用级联注意力 | 开关参数 | `vllm serve model --disable-cascade-attn` | 禁用注意力优化策略 |
| `--enable-sleep-mode` | bool | False | 启用睡眠模式 | 开关参数 | `vllm serve model --enable-sleep-mode` | 空闲时释放 GPU 资源 |
| `--model-impl` | str | "auto" | 模型实现方式 | auto/vllm/transformers | `vllm serve model --model-impl transformers` | 使用 Transformers 后端而非 vLLM 原生实现 |
| `--override-attention-dtype` | str | None | 覆盖注意力计算的数据类型 | float16/bfloat16/float32 | `vllm serve model --override-attention-dtype float32` | 强制注意力使用 float32 精度 |
| `--generation-config` | str | "auto" | 生成配置来源 | auto/[path] | `vllm serve model --generation-config /path/to/gen_config.json` | 使用自定义生成配置 |
| `--override-generation-config` | JSON | None | 覆盖生成配置参数 | JSON 格式 | `vllm serve model --override-generation-config '{"temperature": 0.7}'` | 覆盖默认的生成温度参数 |
| `--skip-tokenizer-init` | bool | False | 跳过分词器初始化 | 开关参数 | `vllm serve model --skip-tokenizer-init` | 在不需要分词器的场景下加速启动 |
| `--enable-prompt-embeds` | bool | False | 启用 prompt embedding 输入 | 开关参数 | `vllm serve model --enable-prompt-embeds` | 允许直接传入 embedding 向量而非文本 |
| `--language-model-only` | bool | False | 仅加载语言模型部分 | 开关参数 | `vllm serve model --language-model-only` | 多模态模型中仅加载文本部分 |

#### 2.3.2 多模态配置 (MultiModal)

| 参数 | 类型 | 默认值 | 作用 | 参数值说明 | 示例 | 示例功能 |
|------|------|--------|------|-----------|------|----------|
| `--limit-mm-per-prompt` | JSON | None | 限制每个 prompt 的多模态输入数量 | JSON 格式 `{"image": N}` | `vllm serve model --limit-mm-per-prompt '{"image": 4}'` | 每个请求最多处理 4 张图片 |
| `--allowed-local-media-path` | str | "" | 允许的本地媒体文件路径 | 目录路径 | `vllm serve model --allowed-local-media-path /data/images` | 允许从指定目录加载本地图片 |
| `--allowed-media-domains` | list | None | 允许的媒体 URL 域名 | JSON 格式域名列表 | `vllm serve model --allowed-media-domains '["cdn.example.com"]'` | 仅允许从指定域名加载远程媒体 |
| `--mm-processor-kwargs` | JSON | None | 多模态处理器额外参数 | JSON 格式 | `vllm serve model --mm-processor-kwargs '{"max_image_size": 512}'` | 设置图像处理的最大尺寸 |
| `--mm-processor-cache-gb` | float | 0 | 多模态处理器缓存大小 (GB) | 非负浮点数 | `vllm serve model --mm-processor-cache-gb 4` | 分配 4GB 缓存给多模态处理器 |
| `--enable-mm-embeds` | bool | False | 启用多模态 embedding | 开关参数 | `vllm serve model --enable-mm-embeds` | 允许输入多模态 embedding |
| `--mm-encoder-only` | bool | False | 仅使用多模态编码器 | 开关参数 | `vllm serve model --mm-encoder-only` | 仅运行视觉编码器部分 |
| `--video-pruning-rate` | float | None | 视频帧修剪率 | 0-1 之间的浮点数 | `vllm serve model --video-pruning-rate 0.5` | 修剪 50% 的视频帧以提高效率 |

#### 2.3.3 并行计算配置 (ParallelConfig)

| 参数 | 类型 | 默认值 | 作用 | 参数值说明 | 示例 | 示例功能 |
|------|------|--------|------|-----------|------|----------|
| `--tensor-parallel-size` / `-tp` | int | 1 | 张量并行大小 | 正整数（通常等于 GPU 数量） | `vllm serve model -tp 4` | 将模型分割到 4 个 GPU 上并行计算 |
| `--pipeline-parallel-size` / `-pp` | int | 1 | 流水线并行大小 | 正整数 | `vllm serve model -pp 2 -tp 2` | 2 级流水线 × 2 路张量并行 = 4 GPU |
| `--data-parallel-size` / `-dp` | int | 1 | 数据并行大小 | 正整数 | `vllm serve model -dp 2 -tp 2` | 2 个数据并行副本，每个使用 2 GPU |
| `--data-parallel-rank` | int | None | 数据并行中当前节点的 rank | 非负整数 | `vllm serve model -dp 2 --data-parallel-rank 1` | 指定此进程为数据并行的第 2 个节点 |
| `--data-parallel-start-rank` | int | None | 混合负载均衡的起始 rank | 非负整数 | `vllm serve model --data-parallel-start-rank 0` | 设置混合 LB 模式的起始 rank |
| `--data-parallel-size-local` | int | None | 本地数据并行大小 | 正整数 | `vllm serve model --data-parallel-size-local 2` | 本节点上运行 2 个引擎副本 |
| `--data-parallel-address` | str | None | 数据并行主节点地址 | IP 地址 | `vllm serve model --data-parallel-address 192.168.1.100` | 指定数据并行的协调节点地址 |
| `--data-parallel-rpc-port` | int | None | 数据并行 RPC 端口 | 端口号 | `vllm serve model --data-parallel-rpc-port 29500` | 指定数据并行通信端口 |
| `--data-parallel-hybrid-lb` | bool | False | 启用混合负载均衡 | 开关参数 | `vllm serve model --data-parallel-hybrid-lb` | 结合内外部负载均衡策略 |
| `--data-parallel-external-lb` | bool | False | 启用外部负载均衡 | 开关参数 | `vllm serve model --data-parallel-external-lb` | 使用外部负载均衡器分发请求 |
| `--distributed-executor-backend` | str | None | 分布式执行后端 | mp/ray/uni | `vllm serve model --distributed-executor-backend ray -tp 4` | 使用 Ray 框架进行分布式推理 |
| `--master-addr` | str | "" | 分布式通信主节点地址 | IP 地址 | `vllm serve model --master-addr 10.0.0.1` | 设置 torch.distributed 主节点 |
| `--master-port` | int | 0 | 分布式通信主节点端口 | 端口号 | `vllm serve model --master-port 29500` | 设置 torch.distributed 端口 |
| `--nnodes` | int | 1 | 节点数量 | 正整数 | `vllm serve model --nnodes 2 -tp 4` | 在 2 个节点上分布式运行 |
| `--node-rank` | int | 0 | 当前节点的 rank | 非负整数 | `vllm serve model --node-rank 1` | 标记当前节点为第 2 个节点 |
| `--distributed-timeout-seconds` | int | None | 分布式操作超时时间(秒) | 正整数 | `vllm serve model --distributed-timeout-seconds 300` | 设置分布式操作 5 分钟超时 |
| `--max-parallel-loading-workers` | int | None | 并行加载模型的 worker 数 | 正整数 | `vllm serve model --max-parallel-loading-workers 4` | 使用 4 个并发 worker 加载模型权重 |
| `--disable-custom-all-reduce` | bool | False | 禁用自定义 AllReduce 通信 | 开关参数 | `vllm serve model --disable-custom-all-reduce` | 回退到 NCCL 默认 AllReduce 实现 |
| `--enable-expert-parallel` | bool | False | 启用专家并行 (MoE 模型) | 开关参数 | `vllm serve model --enable-expert-parallel -tp 4` | 在多 GPU 上并行化 MoE 专家 |
| `--enable-elastic-ep` | bool | False | 启用弹性专家并行 | 开关参数 | `vllm serve model --enable-elastic-ep` | 动态调整专家并行度 |
| `--worker-cls` | str | "" | 自定义 worker 类 | Python 类导入路径 | `vllm serve model --worker-cls my_pkg.MyWorker` | 使用自定义 worker 实现 |
| `--worker-extension-cls` | str | "" | Worker 扩展类 | Python 类导入路径 | `vllm serve model --worker-extension-cls my_pkg.MyExt` | 添加自定义 worker 扩展逻辑 |

#### 2.3.4 Context Parallel 配置

| 参数 | 类型 | 默认值 | 作用 | 参数值说明 | 示例 | 示例功能 |
|------|------|--------|------|-----------|------|----------|
| `--prefill-context-parallel-size` | int | 1 | 预填充阶段上下文并行大小 | 正整数 | `vllm serve model --prefill-context-parallel-size 2` | 预填充时将上下文分割到 2 个 GPU |
| `--decode-context-parallel-size` | int | 1 | 解码阶段上下文并行大小 | 正整数 | `vllm serve model --decode-context-parallel-size 2` | 解码时将上下文分割到 2 个 GPU |

#### 2.3.5 DBO (Disaggregated Batch Orchestration) 配置

| 参数 | 类型 | 默认值 | 作用 | 参数值说明 | 示例 | 示例功能 |
|------|------|--------|------|-----------|------|----------|
| `--enable-dbo` | bool | False | 启用分离式批次编排 | 开关参数 | `vllm serve model --enable-dbo` | 将 prefill 和 decode 分离到不同批次 |
| `--ubatch-size` | int | 0 | 微批次大小 | 非负整数 | `vllm serve model --enable-dbo --ubatch-size 64` | 设置微批次为 64 tokens |
| `--dbo-decode-token-threshold` | int | 0 | DBO 解码 token 阈值 | 非负整数 | `vllm serve model --enable-dbo --dbo-decode-token-threshold 128` | 解码 token 达到 128 时触发批次切换 |
| `--dbo-prefill-token-threshold` | int | 0 | DBO 预填充 token 阈值 | 非负整数 | `vllm serve model --enable-dbo --dbo-prefill-token-threshold 512` | 预填充 token 达到 512 时触发批次切换 |

#### 2.3.6 KV Cache 配置 (CacheConfig)

| 参数 | 类型 | 默认值 | 作用 | 参数值说明 | 示例 | 示例功能 |
|------|------|--------|------|-----------|------|----------|
| `--gpu-memory-utilization` | float | 0.9 | GPU 显存利用率上限 | 0-1 之间的浮点数 | `vllm serve model --gpu-memory-utilization 0.8` | 仅使用 80% 显存（预留显存给其他程序） |
| `--kv-cache-dtype` | str | "auto" | KV Cache 的数据类型 | auto/fp8/fp8_e4m3/fp8_e5m2 | `vllm serve model --kv-cache-dtype fp8` | 使用 FP8 量化 KV Cache（节省 50% 显存） |
| `--block-size` | int | None | KV Cache 块大小 | 正整数（如 16, 32） | `vllm serve model --block-size 16` | 设置 KV Cache 分配块为 16 tokens |
| `--enable-prefix-caching` | bool | None | 启用前缀缓存 (APC) | 开关参数 | `vllm serve model --enable-prefix-caching` | 缓存相同前缀的 KV Cache（加速相似请求） |
| `--num-gpu-blocks-override` | int | None | 手动指定 GPU KV Cache 块数 | 正整数 | `vllm serve model --num-gpu-blocks-override 1000` | 手动分配 1000 个 KV Cache 块 |
| `--kv-cache-memory-bytes` | int | None | KV Cache 内存大小(字节) | 正整数 | `vllm serve model --kv-cache-memory-bytes 4294967296` | 指定 4GB 用于 KV Cache |
| `--calculate-kv-scales` | bool | False | 计算 KV 缩放因子 | 开关参数 | `vllm serve model --calculate-kv-scales --kv-cache-dtype fp8` | FP8 KV Cache 时动态计算缩放因子 |
| `--kv-offloading-size` | float | None | KV Cache CPU 卸载大小 | 浮点数 (GB) | `vllm serve model --kv-offloading-size 8.0` | 将 8GB KV Cache 卸载到 CPU 内存 |

#### 2.3.7 调度器配置 (SchedulerConfig)

| 参数 | 类型 | 默认值 | 作用 | 参数值说明 | 示例 | 示例功能 |
|------|------|--------|------|-----------|------|----------|
| `--max-num-batched-tokens` | int | None | 每个批次的最大 token 数 | 正整数 | `vllm serve model --max-num-batched-tokens 8192` | 每批次最多处理 8192 tokens |
| `--max-num-seqs` | int | None | 每个批次的最大序列数 | 正整数 | `vllm serve model --max-num-seqs 128` | 每批次最多 128 个请求 |
| `--max-num-partial-prefills` | int | 1 | 最大部分预填充数 | 正整数 | `vllm serve model --max-num-partial-prefills 2` | 允许同时进行 2 个分块预填充 |
| `--max-long-partial-prefills` | int | 1 | 最大长序列部分预填充数 | 正整数 | `vllm serve model --max-long-partial-prefills 1` | 限制长序列分块预填充并发数 |
| `--long-prefill-token-threshold` | int | 0 | 长预填充 token 阈值 | 非负整数 | `vllm serve model --long-prefill-token-threshold 2048` | 超过 2048 tokens 的预填充视为长序列 |
| `--scheduling-policy` | str | "fcfs" | 调度策略 | fcfs (先来先服务) / priority | `vllm serve model --scheduling-policy priority` | 使用优先级调度策略 |
| `--scheduler-cls` | str | None | 自定义调度器类 | Python 类导入路径 | `vllm serve model --scheduler-cls my_pkg.MyScheduler` | 使用自定义调度器 |
| `--enable-chunked-prefill` | bool | None | 启用分块预填充 | 开关参数 | `vllm serve model --enable-chunked-prefill` | 将长 prompt 分块处理，减少延迟波动 |
| `--disable-chunked-mm-input` | bool | False | 禁用多模态分块输入 | 开关参数 | `vllm serve model --disable-chunked-mm-input` | 不对多模态输入进行分块 |
| `--stream-interval` | int | 0 | 流式输出间隔 | 非负整数 | `vllm serve model --stream-interval 2` | 每 2 步输出一次流式结果 |

#### 2.3.8 LoRA 适配器配置

| 参数 | 类型 | 默认值 | 作用 | 参数值说明 | 示例 | 示例功能 |
|------|------|--------|------|-----------|------|----------|
| `--enable-lora` | bool | False | 启用 LoRA 适配器支持 | 开关参数 | `vllm serve model --enable-lora` | 允许加载和使用 LoRA 适配器 |
| `--max-loras` | int | 1 | 最大同时加载的 LoRA 数量 | 正整数 | `vllm serve model --enable-lora --max-loras 4` | 同时在 GPU 中保持 4 个 LoRA |
| `--max-lora-rank` | int | 16 | 最大 LoRA 秩 | 正整数（8/16/32/64 等） | `vllm serve model --enable-lora --max-lora-rank 64` | 支持最高 rank=64 的 LoRA |
| `--max-cpu-loras` | int | None | CPU 上缓存的最大 LoRA 数 | 正整数 | `vllm serve model --enable-lora --max-cpu-loras 16` | 在 CPU 缓存 16 个 LoRA 以快速切换 |
| `--lora-dtype` | str | None | LoRA 权重的数据类型 | float16/bfloat16/float32/auto | `vllm serve model --enable-lora --lora-dtype float16` | 以 float16 精度加载 LoRA |
| `--fully-sharded-loras` | bool | False | 完全分片 LoRA | 开关参数 | `vllm serve model --enable-lora --fully-sharded-loras` | 在多 GPU 间完全分片 LoRA 权重 |

#### 2.3.9 模型加载配置 (LoadConfig)

| 参数 | 类型 | 默认值 | 作用 | 参数值说明 | 示例 | 示例功能 |
|------|------|--------|------|-----------|------|----------|
| `--download-dir` | str | None | 模型下载目录 | 目录路径 | `vllm serve model --download-dir /data/models` | 将模型下载到指定目录 |
| `--load-format` | str | "auto" | 模型加载格式 | auto/pt/safetensors/npcache/dummy/bitsandbytes 等 | `vllm serve model --load-format safetensors` | 强制使用 safetensors 格式加载 |
| `--model-loader-extra-config` | JSON | None | 模型加载器额外配置 | JSON 格式 | `vllm serve model --model-loader-extra-config '{}'` | 传递额外配置给模型加载器 |
| `--ignore-patterns` | str/list | None | 加载时忽略的文件模式 | glob 模式 | `vllm serve model --ignore-patterns "*.bin"` | 忽略 .bin 文件，仅加载 safetensors |
| `--use-tqdm-on-load` | bool | True | 加载时显示进度条 | 开关参数 | `vllm serve model --no-use-tqdm-on-load` | 禁用模型加载进度条 |

#### 2.3.10 Offload 卸载配置

| 参数 | 类型 | 默认值 | 作用 | 参数值说明 | 示例 | 示例功能 |
|------|------|--------|------|-----------|------|----------|
| `--cpu-offload-gb` | float | 0 | CPU 内存卸载大小 (GB) | 非负浮点数 | `vllm serve model --cpu-offload-gb 10` | 将 10GB 模型参数卸载到 CPU 内存 |

#### 2.3.11 注意力与内核配置

| 参数 | 类型 | 默认值 | 作用 | 参数值说明 | 示例 | 示例功能 |
|------|------|--------|------|-----------|------|----------|
| `--attention-backend` | str | None | 注意力计算后端 | FLASH_ATTN/FLASHINFER/XFORMERS 等 | `vllm serve model --attention-backend FLASH_ATTN` | 强制使用 FlashAttention 后端 |
| `--moe-backend` | str | "auto" | MoE 层的计算后端 | auto/pplx/deepep_ht/deepep_ll 等 | `vllm serve model --moe-backend pplx` | 使用 PPLX 后端加速 MoE 计算 |
| `--enable-flashinfer-autotune` | bool | False | 启用 FlashInfer 自动调优 | 开关参数 | `vllm serve model --enable-flashinfer-autotune` | 自动选择最优的 FlashInfer 配置 |

#### 2.3.12 Speculative Decoding (投机解码)

| 参数 | 类型 | 默认值 | 作用 | 参数值说明 | 示例 | 示例功能 |
|------|------|--------|------|-----------|------|----------|
| `--speculative-config` | JSON | None | 投机解码配置 | JSON 格式 | `vllm serve model --speculative-config '{"model": "draft_model", "num_speculative_tokens": 5}'` | 使用 draft model 进行投机解码，加速推理 |

#### 2.3.13 可观测性配置 (ObservabilityConfig)

| 参数 | 类型 | 默认值 | 作用 | 参数值说明 | 示例 | 示例功能 |
|------|------|--------|------|-----------|------|----------|
| `--otlp-traces-endpoint` | str | None | OpenTelemetry traces 端点 | URL 地址 | `vllm serve model --otlp-traces-endpoint http://jaeger:4318` | 将追踪数据发送到 Jaeger |
| `--collect-detailed-traces` | list | None | 收集详细追踪模块 | 模块名列表 | `vllm serve model --collect-detailed-traces model,worker` | 收集模型和 worker 的详细追踪信息 |
| `--show-hidden-metrics-for-version` | str | None | 显示指定版本的隐藏指标 | 版本号字符串 | `vllm serve model --show-hidden-metrics-for-version "0.8.0"` | 显示该版本新增的隐藏 Prometheus 指标 |
| `--kv-cache-metrics` | bool | False | 启用 KV Cache 指标 | 开关参数 | `vllm serve model --kv-cache-metrics` | 暴露 KV Cache 使用率等 Prometheus 指标 |
| `--enable-mfu-metrics` | bool | False | 启用 MFU (Model FLOPs Utilization) 指标 | 开关参数 | `vllm serve model --enable-mfu-metrics` | 跟踪模型浮点运算利用率 |

#### 2.3.14 编译与优化配置

| 参数 | 类型 | 默认值 | 作用 | 参数值说明 | 示例 | 示例功能 |
|------|------|--------|------|-----------|------|----------|
| `--compilation-config` | JSON | None | torch.compile 编译配置 | JSON 格式 | `vllm serve model --compilation-config '{"level": 3}'` | 启用 level 3 编译优化 |
| `--max-cudagraph-capture-size` | int | None | CUDA Graph 最大捕获批次大小 | 正整数 | `vllm serve model --max-cudagraph-capture-size 32` | CUDA Graph 最多捕获 batch_size=32 |
| `--optimization-level` | int | 0 | 优化级别 | 0-3 | `vllm serve model --optimization-level 2` | 启用中等优化以平衡速度和兼容性 |

#### 2.3.15 结构化输出与推理配置

| 参数 | 类型 | 默认值 | 作用 | 参数值说明 | 示例 | 示例功能 |
|------|------|--------|------|-----------|------|----------|
| `--reasoning-parser` | str | None | 推理链解析器 | 解析器名称 | `vllm serve model --reasoning-parser deepseek_r1` | 使用 DeepSeek-R1 格式解析推理链 |
| `--reasoning-parser-plugin` | str | None | 自定义推理链解析器插件 | Python 模块路径 | `vllm serve model --reasoning-parser-plugin my_plugin` | 加载自定义推理链解析插件 |

#### 2.3.16 其他引擎参数

| 参数 | 类型 | 默认值 | 作用 | 参数值说明 | 示例 | 示例功能 |
|------|------|--------|------|-----------|------|----------|
| `--shutdown-timeout` | int | 0 | 关闭超时时间(秒) | 非负整数；0=立即终止 | `vllm serve model --shutdown-timeout 30` | 关闭时等待最多 30 秒完成进行中的请求 |
| `--additional-config` | JSON | None | 额外自定义配置 | JSON 格式 | `vllm serve model --additional-config '{"custom_key": "value"}'` | 传递自定义配置参数 |
| `--fail-on-environ-validation` | bool | False | 环境验证失败时抛出错误 | 开关参数 | `vllm serve model --fail-on-environ-validation` | 严格模式：环境不符合要求时直接报错 |

---

## 3. vllm chat - 交互式对话

**作用**：连接到运行中的 vLLM API 服务器，进行交互式对话。

**基本用法**：`vllm chat [options]`

| 参数 | 类型 | 默认值 | 作用 | 参数值说明 | 示例 | 示例功能 |
|------|------|--------|------|-----------|------|----------|
| `--url` | str | "http://localhost:8000/v1" | API 服务器的 URL 地址 | 完整的 API 基础 URL | `vllm chat --url http://192.168.1.100:8000/v1` | 连接到远程服务器进行对话 |
| `--model-name` | str | None | 模型名称 | 字符串；默认自动获取第一个模型 | `vllm chat --model-name "my-model"` | 指定使用名为 "my-model" 的模型 |
| `--api-key` | str | None | API 密钥 | 密钥字符串 | `vllm chat --api-key "sk-mykey"` | 使用密钥认证访问 API |
| `--system-prompt` | str | None | 系统提示词 | 任意文本字符串 | `vllm chat --system-prompt "你是一个中文助手"` | 设置对话的系统提示词 |
| `-q` / `--quick` | str | None | 发送单条消息并退出 | 消息文本 | `vllm chat -q "什么是机器学习？"` | 快速提问并获取回答，不进入交互模式 |

---

## 4. vllm complete - 文本补全

**作用**：连接到运行中的 vLLM API 服务器，进行文本补全。

**基本用法**：`vllm complete [options]`

| 参数 | 类型 | 默认值 | 作用 | 参数值说明 | 示例 | 示例功能 |
|------|------|--------|------|-----------|------|----------|
| `--url` | str | "http://localhost:8000/v1" | API 服务器的 URL 地址 | 完整的 API 基础 URL | `vllm complete --url http://localhost:8000/v1` | 连接到本地 API 服务器 |
| `--model-name` | str | None | 模型名称 | 字符串；默认自动获取 | `vllm complete --model-name "my-model"` | 使用指定模型进行补全 |
| `--api-key` | str | None | API 密钥 | 密钥字符串 | `vllm complete --api-key "sk-mykey"` | 使用密钥认证 |
| `--max-tokens` | int | None | 每个输出序列的最大生成 token 数 | 正整数 | `vllm complete --max-tokens 256` | 限制输出最多 256 tokens |
| `-q` / `--quick` | str | None | 发送单个 prompt 并退出 | prompt 文本 | `vllm complete -q "def fibonacci(n):"` | 快速补全代码并退出 |

---

## 5. vllm run-batch - 批量推理

**作用**：使用 vLLM 引擎批量处理 JSONL 格式的请求文件。

**基本用法**：`vllm run-batch -i INPUT.jsonl -o OUTPUT.jsonl --model <model>`

### 5.1 Batch 专属参数

| 参数 | 类型 | 默认值 | 作用 | 参数值说明 | 示例 | 示例功能 |
|------|------|--------|------|-----------|------|----------|
| `-i` / `--input-file` | str | 必填 | 输入文件路径或 URL | 本地路径或 HTTP URL | `vllm run-batch -i requests.jsonl -o output.jsonl --model model` | 从本地 JSONL 文件读取批量请求 |
| `-o` / `--output-file` | str | 必填 | 输出文件路径或 URL | 本地路径或 HTTP URL | `vllm run-batch -i input.jsonl -o https://storage.example.com/output.jsonl --model model` | 将结果上传到远程存储 |
| `--output-tmp-dir` | str | None | 输出临时目录 | 目录路径 | `vllm run-batch -i in.jsonl -o out.jsonl --output-tmp-dir /tmp/batch --model model` | 先写入临时目录，完成后再移动/上传 |
| `--enable-metrics` | bool | False | 启用 Prometheus 指标 | 开关参数 | `vllm run-batch -i in.jsonl -o out.jsonl --enable-metrics --model model` | 批处理期间暴露 Prometheus 指标 |
| `--host` | str | None | 指标服务器主机地址 | IP 地址 | `vllm run-batch --enable-metrics --host 0.0.0.0 -i in.jsonl -o out.jsonl --model model` | 指标服务器监听所有接口 |
| `--port` | int | 8000 | 指标服务器端口 | 端口号 | `vllm run-batch --enable-metrics --port 9090 -i in.jsonl -o out.jsonl --model model` | 在 9090 端口暴露指标 |

> **注意**：`vllm run-batch` 还继承了 BaseFrontendArgs（2.2.8-2.2.10 节中除网络/SSL/CORS 之外的参数）和全部 Engine 引擎参数（2.3 节）。

---

## 6. vllm launch render - 无GPU渲染服务

**作用**：启动无 GPU 的渲染服务器，仅做预处理和后处理（用于分离式架构）。

**基本用法**：`vllm launch render [model_tag] [options]`

> 参数与 `vllm serve` 基本相同，继承了 FrontendArgs 和 AsyncEngineArgs 的所有参数。

---

## 7. vllm bench - 性能基准测试

**作用**：运行各种性能基准测试。

### 7.1 vllm bench latency - 延迟基准测试

**基本用法**：`vllm bench latency --model <model> [options]`

| 参数 | 类型 | 默认值 | 作用 | 参数值说明 | 示例 | 示例功能 |
|------|------|--------|------|-----------|------|----------|
| `--input-len` | int | 32 | 输入 prompt 长度 | 正整数 (token 数) | `vllm bench latency --model model --input-len 128` | 测试输入 128 tokens 时的延迟 |
| `--output-len` | int | 128 | 输出长度 | 正整数 (token 数) | `vllm bench latency --model model --output-len 256` | 测试生成 256 tokens 的延迟 |
| `--batch-size` | int | 8 | 批次大小 | 正整数 | `vllm bench latency --model model --batch-size 32` | 测试批次 32 时的延迟 |
| `--n` | int | 1 | 每个 prompt 生成的序列数 | 正整数 | `vllm bench latency --model model --n 4` | 每个 prompt 生成 4 个候选 |
| `--use-beam-search` | bool | False | 使用 beam search | 开关参数 | `vllm bench latency --model model --use-beam-search` | 测试 beam search 模式的延迟 |
| `--num-iters-warmup` | int | 10 | 预热迭代次数 | 正整数 | `vllm bench latency --model model --num-iters-warmup 5` | 预热 5 次后再开始计时 |
| `--num-iters` | int | 30 | 测试迭代次数 | 正整数 | `vllm bench latency --model model --num-iters 50` | 运行 50 次迭代取平均 |
| `--profile` | bool | False | 启用性能分析 | 开关参数 | `vllm bench latency --model model --profile` | 生成性能分析报告 |
| `--output-json` | str | None | 结果保存路径 | JSON 文件路径 | `vllm bench latency --model model --output-json result.json` | 将结果保存为 JSON 文件 |
| `--disable-detokenize` | bool | False | 跳过反分词 | 开关参数 | `vllm bench latency --model model --disable-detokenize` | 测试不包含反分词开销的纯延迟 |

> 还支持所有 EngineArgs（2.3 节）。

### 7.2 vllm bench throughput - 吞吐量基准测试

**基本用法**：`vllm bench throughput --model <model> [options]`

| 参数 | 类型 | 默认值 | 作用 | 参数值说明 | 示例 | 示例功能 |
|------|------|--------|------|-----------|------|----------|
| `--backend` | str | "vllm" | 后端引擎类型 | vllm/hf/mii/vllm-chat | `vllm bench throughput --model model --backend hf` | 使用 HuggingFace 原生推理做对比 |
| `--dataset-name` | str | "sharegpt" | 数据集类型 | sharegpt/random/sonnet/burstgpt/hf 等 | `vllm bench throughput --model model --dataset-name random` | 使用随机生成的数据集测试 |
| `--dataset-path` | str | None | 数据集文件路径 | 文件路径或 HF 数据集名 | `vllm bench throughput --model model --dataset-path data.json` | 使用自定义数据集 |
| `--num-prompts` | int | 1000 | 提示词数量 | 正整数 | `vllm bench throughput --model model --num-prompts 500` | 使用 500 条提示词测试 |
| `--input-len` | int | None | 输入长度（random 数据集） | 正整数 | `vllm bench throughput --model model --dataset-name random --input-len 128 --output-len 128` | 测试固定 128 tokens 输入的吞吐量 |
| `--output-len` | int | None | 输出长度（random 数据集） | 正整数 | (同上) | (同上) |
| `--async-engine` | bool | False | 使用异步引擎 | 开关参数 | `vllm bench throughput --model model --async-engine` | 使用异步模式测试吞吐量 |
| `--output-json` | str | None | 结果保存路径 | JSON 文件路径 | `vllm bench throughput --model model --output-json result.json` | 保存吞吐量结果 |

> 还支持所有 AsyncEngineArgs/EngineArgs（2.3 节）。

### 7.3 vllm bench serve - 在线服务基准测试

**基本用法**：`vllm bench serve --model <model> [options]`

| 参数 | 类型 | 默认值 | 作用 | 参数值说明 | 示例 | 示例功能 |
|------|------|--------|------|-----------|------|----------|
| `--backend` | str | "openai" | API 后端类型 | openai/tgi/azure 等 | `vllm bench serve --model model --backend openai` | 测试 OpenAI 兼容 API 性能 |
| `--base-url` | str | None | 服务器基础 URL | HTTP URL | `vllm bench serve --model model --base-url http://localhost:8000` | 连接到指定地址的服务 |
| `--host` | str | "127.0.0.1" | 服务器主机 | IP 地址 | `vllm bench serve --model model --host 0.0.0.0` | 连接到指定 host |
| `--port` | int | 8000 | 服务器端口 | 端口号 | `vllm bench serve --model model --port 8080` | 连接到 8080 端口 |
| `--endpoint` | str | "/v1/completions" | API 端点路径 | URL 路径 | `vllm bench serve --model model --endpoint /v1/chat/completions` | 测试 chat completions 端点 |
| `--request-rate` | float | inf | 每秒请求数 | 正浮点数或 inf | `vllm bench serve --model model --request-rate 10` | 以每秒 10 个请求发送 |
| `--max-concurrency` | int | None | 最大并发请求数 | 正整数 | `vllm bench serve --model model --max-concurrency 50` | 最多 50 个并发请求 |
| `--num-prompts` | int | 1000 | 发送的请求总数 | 正整数 | `vllm bench serve --model model --num-prompts 100` | 发送 100 个请求测试 |
| `--save-result` | bool | False | 保存结果到 JSON | 开关参数 | `vllm bench serve --model model --save-result` | 保存基准测试结果 |
| `--save-detailed` | bool | False | 保存每个请求的详细信息 | 开关参数 | `vllm bench serve --model model --save-result --save-detailed` | 保存每个请求的延迟等详细信息 |
| `--percentile-metrics` | str | None | 需要计算百分位的指标 | 逗号分隔的指标名 | `vllm bench serve --model model --percentile-metrics "ttft,tpot"` | 计算首 token 时间和每 token 时间的百分位 |
| `--metric-percentiles` | str | "99" | 百分位值 | 逗号分隔的数值 | `vllm bench serve --model model --metric-percentiles "50,90,99"` | 计算 P50、P90、P99 |
| `--ignore-eos` | bool | False | 忽略 EOS token | 开关参数 | `vllm bench serve --model model --ignore-eos` | 强制生成到 max_tokens 长度 |
| `--temperature` | float | None | 采样温度 | 非负浮点数 | `vllm bench serve --model model --temperature 0.8` | 以温度 0.8 采样 |
| `--top-p` | float | None | Top-p 采样 | 0-1 浮点数 | `vllm bench serve --model model --top-p 0.95` | 使用 nucleus sampling |
| `--plot-timeline` | bool | False | 生成时间线图 | 开关参数 | `vllm bench serve --model model --save-result --plot-timeline` | 生成请求时间线可视化 |

### 7.4 vllm bench startup - 启动时间基准测试

**基本用法**：`vllm bench startup --model <model> [options]`

| 参数 | 类型 | 默认值 | 作用 | 参数值说明 | 示例 | 示例功能 |
|------|------|--------|------|-----------|------|----------|
| `--num-iters-cold` | int | 3 | 冷启动测试次数 | 正整数 | `vllm bench startup --model model --num-iters-cold 5` | 测试 5 次冷启动时间 |
| `--num-iters-warmup` | int | 1 | 预热次数 | 正整数 | `vllm bench startup --model model --num-iters-warmup 2` | 预热 2 次 |
| `--num-iters-warm` | int | 3 | 热启动测试次数 | 正整数 | `vllm bench startup --model model --num-iters-warm 5` | 测试 5 次热启动时间 |
| `--output-json` | str | None | 结果保存路径 | JSON 文件路径 | `vllm bench startup --model model --output-json startup.json` | 保存启动时间结果 |

> 还支持所有 EngineArgs（2.3 节）。

### 7.5 vllm bench mm-processor - 多模态处理器基准测试

**基本用法**：`vllm bench mm-processor --model <model> [options]`

| 参数 | 类型 | 默认值 | 作用 | 参数值说明 | 示例 | 示例功能 |
|------|------|--------|------|-----------|------|----------|
| `--dataset-name` | str | "random-mm" | 数据集类型 | random-mm/hf | `vllm bench mm-processor --model model --dataset-name random-mm` | 使用随机多模态数据测试 |
| `--num-prompts` | int | 10 | 提示词数量 | 正整数 | `vllm bench mm-processor --model model --num-prompts 50` | 处理 50 个多模态提示 |
| `--num-warmups` | int | 1 | 预热次数 | 正整数 | `vllm bench mm-processor --model model --num-warmups 3` | 预热 3 次 |
| `--output-json` | str | None | 结果保存路径 | JSON 文件路径 | `vllm bench mm-processor --model model --output-json mm.json` | 保存多模态处理器性能结果 |
| `--disable-tqdm` | bool | False | 禁用进度条 | 开关参数 | `vllm bench mm-processor --model model --disable-tqdm` | 不显示进度条（CI 场景） |

### 7.6 vllm bench sweep - 参数扫描

**基本用法**：`vllm bench sweep <subcommand> [options]`

支持子命令：
- `serve` - 扫描服务器参数
- `serve_workload` - 扫描工作负载参数
- `startup` - 扫描启动参数
- `plot` - 绘制扫描结果图
- `plot_pareto` - 绘制帕累托前沿图

---

## 8. vllm collect-env - 环境信息收集

**作用**：收集系统环境信息（用于问题排查和 bug 报告）。

**基本用法**：`vllm collect-env`

无额外参数。输出包括：操作系统、Python 版本、PyTorch 版本、CUDA 版本、GPU 信息、已安装的相关包等。

**示例**：
```bash
vllm collect-env
```
**功能**：收集并打印完整的环境信息，方便提交 issue 时附上环境描述。

---

## 快速参考：常用命令示例

```bash
# 基础启动
vllm serve Qwen/Qwen2-7B

# 多 GPU 张量并行
vllm serve meta-llama/Llama-3-70B -tp 4 --gpu-memory-utilization 0.85

# FP8 量化 + 前缀缓存
vllm serve model -q fp8 --enable-prefix-caching --kv-cache-dtype fp8

# 数据并行 + 自定义端口
vllm serve model -dp 2 -tp 2 --port 8080

# 加载 LoRA 适配器
vllm serve model --enable-lora --lora-modules my_lora=/path/to/lora --max-loras 4

# 工具调用支持
vllm serve model --enable-auto-tool-choice --tool-call-parser hermes

# 批量推理
vllm run-batch -i requests.jsonl -o results.jsonl --model model

# 延迟测试
vllm bench latency --model model --input-len 128 --output-len 256 --batch-size 16

# 在线服务基准测试
vllm bench serve --model model --request-rate 10 --num-prompts 1000

# 快速对话
vllm chat -q "Hello, what is vLLM?"

# 环境信息
vllm collect-env
```
