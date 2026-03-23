# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# [测试通过插件机制进行的树外模型注册：验证自定义模型能通过 API 正常推理]

from ...utils import VLLM_PATH, RemoteOpenAIServer

chatml_jinja_path = VLLM_PATH / "examples/template_chatml.jinja"
assert chatml_jinja_path.exists()


# [启动自定义注册的 OPT 模型服务器并验证聊天补全]
def run_and_test_dummy_opt_api_server(model, tp=1):
    # the model is registered through the plugin
    server_args = [
        "--gpu-memory-utilization",
        "0.10",
        "--dtype",
        "float32",
        "--chat-template",
        str(chatml_jinja_path),
        "--load-format",
        "dummy",
        "-tp",
        f"{tp}",
    ]
    with RemoteOpenAIServer(model, server_args) as server:
        client = server.get_client()
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
            temperature=0,
        )
        generated_text = completion.choices[0].message.content
        assert generated_text is not None
        # make sure only the first token is generated
        rest = generated_text.replace("<s>", "")
        assert rest == ""


# [测试树外注册模型能否通过 API 服务器正常运行]
def test_oot_registration_for_api_server(dummy_opt_path: str):
    run_and_test_dummy_opt_api_server(dummy_opt_path)
