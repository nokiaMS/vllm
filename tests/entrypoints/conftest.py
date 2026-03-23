# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# 入口点测试的共享 pytest fixture 配置文件，提供测试用的提示文本、token ID、
# JSON schema、正则表达式、LoRA 文件等公共测试数据

import pytest


# 提供一组示例文本提示，用于生成类测试
@pytest.fixture
def sample_prompts():
    return [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]


# 提供一组示例 token ID 列表，用于基于 token 的生成测试
@pytest.fixture
def sample_token_ids():
    return [
        [0],
        [0, 1],
        [0, 2, 1],
        [0, 3, 1, 2],
    ]


# 提供一个 IP 地址格式的正则表达式，用于结构化输出测试
@pytest.fixture
def sample_regex():
    return (
        r"((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)\.){3}"
        r"(25[0-5]|(2[0-4]|1\d|[1-9]|)\d)"
    )


# 提供一个包含嵌套对象和数组的 JSON schema，用于结构化输出验证测试
@pytest.fixture
def sample_json_schema():
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "skills": {
                "type": "array",
                "items": {"type": "string", "maxLength": 10},
                "minItems": 3,
            },
            "work_history": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "company": {"type": "string"},
                        "duration": {"type": "number"},
                        "position": {"type": "string"},
                    },
                    "required": ["company", "position"],
                },
            },
        },
        "required": ["name", "age", "skills", "work_history"],
    }


# 提供一个包含数值范围、正则模式等复杂约束的 JSON schema
@pytest.fixture
def sample_complex_json_schema():
    return {
        "type": "object",
        "properties": {
            "score": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,  # Numeric range
            },
            "grade": {
                "type": "string",
                "pattern": "^[A-D]$",  # Regex pattern
            },
            "email": {
                "type": "string",
                "pattern": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$",
            },
            "tags": {
                "type": "array",
                "items": {
                    "type": "string",
                    # Combining length and pattern restrictions
                    "pattern": "^[a-z]{1,10}$",
                },
            },
        },
        "required": ["score", "grade", "email", "tags"],
    }


# 提供一个使用 $defs 引用定义的 JSON schema，用于测试递归/引用类型
@pytest.fixture
def sample_definition_json_schema():
    return {
        "$defs": {
            "Step": {
                "properties": {
                    "explanation": {"title": "Explanation", "type": "string"},
                    "output": {"title": "Output", "type": "string"},
                },
                "required": ["explanation", "output"],
                "title": "Step",
                "type": "object",
            }
        },
        "properties": {
            "steps": {
                "items": {"$ref": "#/$defs/Step"},
                "title": "Steps",
                "type": "array",
            },
            "final_answer": {"title": "Final Answer", "type": "string"},
        },
        "required": ["steps", "final_answer"],
        "title": "MathReasoning",
        "type": "object",
    }


# 提供一个包含枚举约束的 JSON schema，用于测试受限值输出
@pytest.fixture
def sample_enum_json_schema():
    return {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["active", "inactive", "pending"],  # Literal values using enum
            },
            "priority": {
                "type": "string",
                "enum": ["low", "medium", "high", "critical"],
            },
            "category": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["bug", "feature", "improvement"],
                    },
                    "severity": {
                        "type": "integer",
                        "enum": [1, 2, 3, 4, 5],  # Enum can also contain numbers
                    },
                },
                "required": ["type", "severity"],
            },
            "flags": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["urgent", "blocked", "needs_review", "approved"],
                },
            },
        },
        "required": ["status", "priority", "category", "flags"],
    }


# 提供一组编程语言名称选项，用于结构化输出选择测试
@pytest.fixture
def sample_structured_outputs_choices():
    return [
        "Python",
        "Java",
        "JavaScript",
        "C++",
        "C#",
        "PHP",
        "TypeScript",
        "Ruby",
        "Swift",
        "Kotlin",
    ]


# 提供一个 SQL 语法规则字符串，用于测试语法引导的生成
@pytest.fixture
def sample_sql_statements():
    return """
start: select_statement
select_statement: "SELECT" column "from" table "where" condition
column: "col_1" | "col_2"
table: "table_1" | "table_2"
condition: column "=" number
number: "1" | "2"
"""


# 下载 Qwen3 自我认知 LoRA 权重文件（每个测试会话仅下载一次）
@pytest.fixture(scope="session")
def qwen3_lora_files():
    """Download Qwen3 LoRA files once per test session."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id="charent/self_cognition_Alice")


# 下载 Qwen3 "Meow" LoRA 权重文件（每个测试会话仅下载一次）
@pytest.fixture(scope="session")
def qwen3_meowing_lora_files():
    """Download Qwen3 LoRA files once per test session."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id="Jackmin108/Qwen3-0.6B-Meow-LoRA")


# 下载 Qwen3 "Woof" LoRA 权重文件（每个测试会话仅下载一次）
@pytest.fixture(scope="session")
def qwen3_woofing_lora_files():
    """Download Qwen3 LoRA files once per test session."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id="Jackmin108/Qwen3-0.6B-Woof-LoRA")


# 下载 OPT-125M LoRA 权重文件（每个测试会话仅下载一次）
@pytest.fixture(scope="session")
def opt125_lora_files() -> str:
    """Download opt-125m LoRA files once per test session."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id="peft-internal-testing/opt-125m-dummy-lora")
