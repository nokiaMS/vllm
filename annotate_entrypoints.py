#!/usr/bin/env python3
"""
Script to add Chinese comments to all Python files in vllm/entrypoints/.
Rules:
- Add Chinese inline comments (#) to every line of code
- Add Chinese docstrings/comments for every class
- Add Chinese docstrings/comments for every function
- Add Chinese descriptions for import statements
- If a line already has Chinese comments, do NOT duplicate
- Do NOT delete or modify any original content
- Empty lines and pure comment lines don't need annotation
- Skip empty __init__.py files (only license header)
"""

import os
import re
import ast
import textwrap

# Map of common import modules to Chinese descriptions
IMPORT_DESCRIPTIONS = {
    # Standard library
    "os": "操作系统接口模块",
    "sys": "系统相关参数和函数",
    "json": "JSON编解码模块",
    "asyncio": "异步I/O框架",
    "signal": "信号处理模块",
    "socket": "网络套接字模块",
    "ssl": "SSL/TLS安全套接字模块",
    "time": "时间相关函数",
    "logging": "日志记录模块",
    "argparse": "命令行参数解析模块",
    "dataclasses": "数据类装饰器和函数",
    "functools": "高阶函数和可调用对象操作工具",
    "itertools": "迭代器工具模块",
    "pathlib": "面向对象的文件系统路径",
    "typing": "类型提示支持模块",
    "warnings": "警告控制模块",
    "abc": "抽象基类模块",
    "collections": "容器数据类型模块",
    "enum": "枚举类型模块",
    "math": "数学函数模块",
    "copy": "浅拷贝和深拷贝操作",
    "io": "I/O核心工具模块",
    "re": "正则表达式模块",
    "uuid": "UUID生成模块",
    "http": "HTTP模块",
    "string": "字符串操作模块",
    "textwrap": "文本换行和填充模块",
    "contextlib": "上下文管理器工具模块",
    "inspect": "运行时对象信息获取模块",
    "struct": "字节与C结构体转换模块",
    "tempfile": "临时文件和目录创建模块",
    "traceback": "异常回溯模块",
    "base64": "Base64编解码模块",
    "hashlib": "安全哈希和消息摘要模块",
    "hmac": "HMAC消息认证码模块",
    "subprocess": "子进程管理模块",
    "threading": "线程模块",
    "multiprocessing": "多进程模块",
    "concurrent": "并发执行模块",
    "queue": "同步队列模块",
    "weakref": "弱引用模块",
    "array": "高效数值数组模块",
    "bisect": "二分查找模块",
    "heapq": "堆队列算法模块",
    "operator": "标准运算符函数模块",
    "pprint": "格式化输出模块",
    "decimal": "十进制浮点运算模块",
    "fractions": "有理数模块",
    "random": "随机数生成模块",
    "statistics": "统计函数模块",
    "csv": "CSV文件读写模块",
    "configparser": "配置文件解析模块",
    "xml": "XML处理模块",
    "html": "HTML处理模块",
    "urllib": "URL处理模块",
    "gzip": "gzip压缩模块",
    "zipfile": "ZIP压缩模块",
    "tarfile": "tar归档模块",
    "shutil": "高级文件操作模块",
    "glob": "文件名模式匹配模块",
    "fnmatch": "Unix文件名模式匹配模块",
    "stat": "文件状态模块",
    "fileinput": "逐行读取多文件模块",
    "pickle": "对象序列化模块",
    "shelve": "对象持久化模块",
    "sqlite3": "SQLite数据库模块",
    "zlib": "压缩模块",
    "lzma": "LZMA压缩模块",
    "bz2": "bz2压缩模块",
    "codecs": "编解码器注册模块",
    "unicodedata": "Unicode字符数据库模块",
    "locale": "本地化模块",
    "gettext": "国际化模块",
    "secrets": "安全随机数模块",
    "platform": "平台信息模块",
    "ctypes": "外部C函数库调用模块",
    "importlib": "导入机制工具模块",
    "pkgutil": "包工具模块",
    "types": "动态类型创建和内置类型名称模块",
    "errno": "系统错误码模块",
    "select": "I/O多路复用模块",
    "selectors": "高级I/O多路复用模块",
    "mmap": "内存映射文件模块",
    "sched": "事件调度器模块",
    "datetime": "日期时间模块",
    "calendar": "日历模块",
    "atexit": "退出处理程序模块",
    "dis": "Python字节码反汇编器模块",
    "gc": "垃圾收集器模块",
    "resource": "资源使用信息模块",
    "tracemalloc": "内存分配跟踪模块",

    # Collections sub-modules
    "collections.abc": "容器抽象基类",
    "collections.OrderedDict": "有序字典",
    "collections.defaultdict": "带默认值的字典",
    "collections.Counter": "计数器",
    "collections.deque": "双端队列",

    # Typing sub-modules
    "typing.TYPE_CHECKING": "类型检查标志",
    "typing_extensions": "类型提示扩展模块",

    # Third party - web
    "fastapi": "FastAPI高性能Web框架",
    "uvicorn": "ASGI服务器",
    "starlette": "轻量级ASGI框架",
    "pydantic": "数据验证和设置管理库",
    "grpc": "gRPC远程过程调用框架",
    "aiohttp": "异步HTTP客户端/服务器",
    "requests": "HTTP请求库",
    "httpx": "现代HTTP客户端",
    "websockets": "WebSocket库",

    # Third party - ML
    "torch": "PyTorch深度学习框架",
    "numpy": "NumPy数值计算库",
    "transformers": "HuggingFace Transformers库",
    "PIL": "Python图像处理库",
    "tqdm": "进度条库",
    "cloudpickle": "扩展的pickle序列化库",

    # Third party - misc
    "regex": "增强正则表达式模块",
    "openai": "OpenAI API客户端库",
    "openai_harmony": "OpenAI兼容消息协调库",
    "uvloop": "高性能事件循环库",
    "watchfiles": "文件变更监控库",
    "prometheus_client": "Prometheus监控客户端",
    "msgspec": "高性能消息序列化库",
    "lark": "语法解析器库",

    # vLLM modules
    "vllm": "vLLM大语言模型推理引擎",
    "vllm.envs": "vLLM环境变量配置",
    "vllm.config": "vLLM配置模块",
    "vllm.logger": "vLLM日志模块",
    "vllm.engine": "vLLM引擎模块",
    "vllm.sampling_params": "采样参数模块",
    "vllm.pooling_params": "池化参数模块",
    "vllm.utils": "vLLM工具模块",
    "vllm.lora": "LoRA适配器模块",
    "vllm.outputs": "输出数据结构模块",
    "vllm.sequence": "序列数据模块",
    "vllm.inputs": "输入数据模块",
    "vllm.multimodal": "多模态处理模块",
    "vllm.platforms": "平台适配模块",
    "vllm.version": "版本信息模块",
    "vllm.usage": "使用统计模块",
    "vllm.tasks": "任务定义模块",
    "vllm.tokenizers": "分词器模块",
    "vllm.renderers": "渲染器模块",
    "vllm.v1": "vLLM v1引擎模块",
    "vllm.entrypoints": "vLLM入口点模块",
    "vllm.distributed": "分布式处理模块",
    "vllm.model_executor": "模型执行器模块",
    "vllm.transformers_utils": "Transformers工具模块",
    "vllm.beam_search": "束搜索模块",
    "vllm.forward_context": "前向传播上下文模块",
    "vllm.connections": "连接管理模块",
    "vllm.exceptions": "异常定义模块",
    "vllm.logits_process": "logits处理模块",
    "vllm.logprobs": "对数概率模块",
    "vllm.scalar_type": "标量类型模块",
    "vllm.tracing": "追踪模块",
}

# Chinese descriptions for common code patterns
CODE_PATTERNS = {
    # Decorators
    r'^\s*@app\.(get|post|put|delete|patch)\(': "注册{method}路由端点",
    r'^\s*@(staticmethod|classmethod)': "{decorator}装饰器",
    r'^\s*@property': "属性装饰器",
    r'^\s*@abstractmethod': "抽象方法装饰器",
    r'^\s*@(overload)': "函数重载装饰器",
    r'^\s*@(dataclass)': "数据类装饰器",
    r'^\s*@(lru_cache|cached_property)': "缓存装饰器",
    r'^\s*@functools\.wraps': "保留原函数元信息的装饰器",
    r'^\s*@router\.(get|post|put|delete|patch|websocket)\(': "注册{method}路由端点",
    r'^\s*@app\.middleware': "注册中间件",

    # Control flow
    r'^\s*if __name__\s*==\s*["\']__main__["\']': "主程序入口判断",
    r'^\s*try\s*:': "异常捕获开始",
    r'^\s*except\s+': "捕获异常",
    r'^\s*except\s*:': "捕获所有异常",
    r'^\s*finally\s*:': "最终执行块",
    r'^\s*else\s*:': "否则分支",
    r'^\s*elif\s+': "否则如果条件分支",
    r'^\s*while\s+': "循环执行",
    r'^\s*for\s+.*\s+in\s+': "遍历循环",
    r'^\s*with\s+': "上下文管理器",
    r'^\s*async\s+with\s+': "异步上下文管理器",
    r'^\s*async\s+for\s+': "异步遍历循环",
    r'^\s*yield\s': "生成器返回值",
    r'^\s*yield\s*$': "生成器暂停",
    r'^\s*return\s+None\s*$': "返回空值",
    r'^\s*return\s*$': "返回空值",
    r'^\s*return\s+': "返回结果",
    r'^\s*raise\s+': "抛出异常",
    r'^\s*pass\s*$': "空操作占位符",
    r'^\s*break\s*$': "跳出循环",
    r'^\s*continue\s*$': "继续下一次循环",

    # Assignments and operations
    r'^\s*assert\s+': "断言检查",
    r'^\s*global\s+': "声明全局变量",
    r'^\s*nonlocal\s+': "声明非局部变量",
    r'^\s*del\s+': "删除对象",
    r'^\s*await\s+': "等待异步操作完成",

    # Logging
    r'^\s*logger\.(info|debug|warning|error|critical|exception)\(': "记录{level}日志",
}


def has_chinese(text):
    """Check if text contains Chinese characters."""
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def is_empty_init_file(filepath, content):
    """Check if file is an empty __init__.py (only license header and maybe empty lines)."""
    basename = os.path.basename(filepath)
    if basename != "__init__.py":
        return False
    lines = content.strip().split('\n')
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('#') and not stripped.startswith('"""') and not stripped.startswith("'''"):
            return False
    return True


def get_import_comment(line):
    """Generate Chinese comment for an import statement."""
    stripped = line.strip()

    # Already has a comment
    if '#' in line:
        code_part = line.split('#')[0]
        comment_part = line.split('#', 1)[1]
        if has_chinese(comment_part):
            return None  # Already has Chinese comment

    # Parse import
    if stripped.startswith('from '):
        match = re.match(r'from\s+([\w.]+)\s+import\s+(.+)', stripped)
        if match:
            module = match.group(1)
            imports = match.group(2).strip()

            # Check for known module
            desc = None
            # Try exact match first
            if module in IMPORT_DESCRIPTIONS:
                desc = IMPORT_DESCRIPTIONS[module]
            else:
                # Try prefix match
                parts = module.split('.')
                for i in range(len(parts), 0, -1):
                    prefix = '.'.join(parts[:i])
                    if prefix in IMPORT_DESCRIPTIONS:
                        desc = IMPORT_DESCRIPTIONS[prefix]
                        break

            if desc:
                return f"从{desc}导入{imports.split(',')[0].strip().split(' ')[0]}"
            else:
                return f"从{module}模块导入相关组件"
    elif stripped.startswith('import '):
        match = re.match(r'import\s+([\w.]+)(?:\s+as\s+(\w+))?', stripped)
        if match:
            module = match.group(1)
            alias = match.group(2)

            desc = IMPORT_DESCRIPTIONS.get(module)
            if not desc:
                parts = module.split('.')
                for i in range(len(parts), 0, -1):
                    prefix = '.'.join(parts[:i])
                    if prefix in IMPORT_DESCRIPTIONS:
                        desc = IMPORT_DESCRIPTIONS[prefix]
                        break

            if desc:
                if alias:
                    return f"导入{desc}(别名{alias})"
                return f"导入{desc}"
            else:
                if alias:
                    return f"导入{module}模块(别名{alias})"
                return f"导入{module}模块"
    return None


def get_class_comment(class_name, bases):
    """Generate Chinese docstring for a class."""
    if bases:
        return f"    \"\"\"类{class_name}: 继承自{', '.join(bases)}的类定义。\"\"\""
    return f"    \"\"\"类{class_name}: 类定义。\"\"\""


def get_function_comment(func_name, is_async=False, is_method=False):
    """Generate Chinese description for a function."""
    prefix = "异步" if is_async else ""
    kind = "方法" if is_method else "函数"
    return f"{prefix}{kind}{func_name}的定义"


def get_code_comment(line):
    """Generate Chinese comment for a code line based on patterns."""
    stripped = line.strip()

    # Skip empty lines, comments-only lines, and lines that are just closing brackets
    if not stripped or stripped.startswith('#') or stripped in (')', ']', '}', '),', '],', '},'):
        return None

    # Skip lines that are only string literals (docstrings)
    if stripped.startswith(('"""', "'''", '"', "'")):
        return None

    # Skip continuation lines (part of multi-line expressions)
    # These are lines that are just arguments, closing parens, etc.
    if stripped.startswith((')', ']', '}')) or stripped == '...':
        return None

    # Check known patterns
    for pattern, template in CODE_PATTERNS.items():
        match = re.match(pattern, line)
        if match:
            groups = match.groups()
            if groups:
                return template.format(method=groups[0], decorator=groups[0] if len(groups) > 0 else '', level=groups[0] if len(groups) > 0 else '')
            return template

    return None


def should_skip_line(line):
    """Determine if a line should not receive a comment."""
    stripped = line.strip()

    # Empty lines
    if not stripped:
        return True

    # Pure comment lines
    if stripped.startswith('#'):
        return True

    # Docstring lines
    if stripped.startswith(('"""', "'''")) or stripped.endswith(('"""', "'''")):
        return True

    # Lines that are part of multi-line strings
    # (This is hard to detect perfectly without full parsing)

    # Lines that are just closing brackets/parens
    if stripped in (')', ']', '}', '),', '],', '},', ')', '...'):
        return True

    # Lines that are just type annotations or continuation
    if stripped.startswith(('|', '->', ':')):
        return True

    return False


def has_inline_comment(line):
    """Check if line already has an inline comment (not in a string)."""
    # Simple check: find # that's not inside a string
    in_single = False
    in_double = False
    in_triple_single = False
    in_triple_double = False
    i = 0
    while i < len(line):
        if line[i:i+3] == '"""' and not in_single and not in_triple_single:
            in_triple_double = not in_triple_double
            i += 3
            continue
        if line[i:i+3] == "'''" and not in_double and not in_triple_double:
            in_triple_single = not in_triple_single
            i += 3
            continue
        if not in_triple_single and not in_triple_double:
            if line[i] == '"' and not in_single:
                in_double = not in_double
            elif line[i] == "'" and not in_double:
                in_single = not in_single
            elif line[i] == '#' and not in_single and not in_double:
                return True
        i += 1
    return False


def annotate_file(filepath):
    """Add Chinese comments to a Python file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Skip empty __init__.py files
    if is_empty_init_file(filepath, content):
        return False

    lines = content.split('\n')
    new_lines = []
    in_multiline_string = False
    multiline_char = None
    in_class = False
    class_indent = 0
    just_saw_class = False
    just_saw_func = False
    func_is_async = False
    func_is_method = False
    func_name = ""
    class_name = ""
    class_bases = []
    # Track if we are inside a multi-line import (parenthesized)
    in_multiline_import = False
    # Track if next non-empty line should have a docstring insertion
    insert_class_docstring = False
    insert_func_docstring = False
    # Track multi-line def or class
    in_multiline_def = False
    paren_depth = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Handle multi-line strings
        if in_multiline_string:
            new_lines.append(line)
            if multiline_char in stripped:
                # Count occurrences to handle opening and closing on different lines
                count = stripped.count(multiline_char)
                if count % 2 == 1:
                    in_multiline_string = False
            i += 1
            continue

        # Check for start of multi-line string
        for mc in ('"""', "'''"):
            count = stripped.count(mc)
            if count == 1:  # Opening but not closing on same line
                in_multiline_string = True
                multiline_char = mc
                break

        if in_multiline_string:
            new_lines.append(line)
            i += 1
            continue

        # Insert Chinese docstring after class definition
        if insert_class_docstring:
            # Check if line is already a docstring
            if stripped.startswith(('"""', "'''", '"', "'")):
                # There's already a docstring, add Chinese after it
                new_lines.append(line)
                # Find end of docstring
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    mc = stripped[:3]
                    if stripped.count(mc) >= 2 and len(stripped) > 3:
                        # Single-line docstring - add Chinese version after
                        indent = ' ' * (class_indent + 4)
                        chinese_doc = f'{indent}# 中文: 类{class_name}'
                        if class_bases:
                            chinese_doc += f'(继承自{", ".join(class_bases)})'
                        new_lines.append(chinese_doc)
                        insert_class_docstring = False
                    else:
                        # Multi-line docstring - will add after closing
                        in_multiline_string = True
                        multiline_char = mc
                        insert_class_docstring = False
                else:
                    insert_class_docstring = False
            elif stripped:
                # No docstring present, add one
                indent = ' ' * (class_indent + 4)
                chinese_doc = f'{indent}# 中文: 类{class_name}'
                if class_bases:
                    chinese_doc += f'(继承自{", ".join(class_bases)})'
                new_lines.append(chinese_doc)
                insert_class_docstring = False
                # Don't increment i, process this line normally
                continue
            else:
                new_lines.append(line)
                i += 1
                continue
            i += 1
            continue

        # Insert Chinese docstring after function definition
        if insert_func_docstring:
            if stripped.startswith(('"""', "'''", '"', "'")):
                new_lines.append(line)
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    mc = stripped[:3]
                    if stripped.count(mc) >= 2 and len(stripped) > 3:
                        # Single-line docstring
                        indent = ' ' * (class_indent + 4) if func_is_method else '    '
                        kind = "异步" if func_is_async else ""
                        fkind = "方法" if func_is_method else "函数"
                        chinese_doc = f'{indent}# 中文: {kind}{fkind}{func_name}'
                        new_lines.append(chinese_doc)
                        insert_func_docstring = False
                    else:
                        in_multiline_string = True
                        multiline_char = mc
                        insert_func_docstring = False
                else:
                    insert_func_docstring = False
            elif stripped:
                # No docstring, insert comment
                if func_is_method:
                    indent = ' ' * (class_indent + 8)
                else:
                    indent = '        '
                kind = "异步" if func_is_async else ""
                fkind = "方法" if func_is_method else "函数"
                chinese_doc = f'{indent}# 中文: {kind}{fkind}{func_name}的实现'
                new_lines.append(chinese_doc)
                insert_func_docstring = False
                continue
            else:
                new_lines.append(line)
                i += 1
                continue
            i += 1
            continue

        # Handle multi-line def/class continuation
        if in_multiline_def:
            new_lines.append(line)
            paren_depth += stripped.count('(') - stripped.count(')')
            if paren_depth <= 0 or stripped.endswith(':'):
                in_multiline_def = False
                if just_saw_class:
                    insert_class_docstring = True
                    just_saw_class = False
                elif just_saw_func:
                    insert_func_docstring = True
                    just_saw_func = False
            i += 1
            continue

        # Handle multi-line import
        if in_multiline_import:
            new_lines.append(line)
            if ')' in stripped:
                in_multiline_import = False
            i += 1
            continue

        # Empty line
        if not stripped:
            new_lines.append(line)
            i += 1
            continue

        # Pure comment line
        if stripped.startswith('#'):
            new_lines.append(line)
            i += 1
            continue

        # Check for class definition
        class_match = re.match(r'^(\s*)class\s+(\w+)(?:\((.*?)\))?\s*:', stripped) or \
                      re.match(r'^(\s*)class\s+(\w+)(?:\((.*?)\))?\s*:', line)
        if not class_match:
            class_match = re.match(r'^(\s*)class\s+(\w+)', line)

        if class_match or stripped.startswith('class '):
            m = re.match(r'^(\s*)class\s+(\w+)(?:\(([^)]*)\))?\s*:?\s*$', line)
            if m:
                class_indent = len(m.group(1))
                class_name = m.group(2)
                bases_str = m.group(3) or ""
                class_bases = [b.strip() for b in bases_str.split(',') if b.strip()] if bases_str else []
                in_class = True

                # Check if definition is complete (has colon)
                if ':' in stripped:
                    # Add comment to class line
                    if not has_inline_comment(line) or not has_chinese(line):
                        comment = f"定义类{class_name}"
                        if class_bases:
                            comment += f"(继承自{', '.join(class_bases[:2])})"
                        if has_inline_comment(line):
                            if not has_chinese(line.split('#', 1)[1]):
                                new_lines.append(f"{line}  # {comment}")
                            else:
                                new_lines.append(line)
                        else:
                            new_lines.append(f"{line}  # {comment}")
                    else:
                        new_lines.append(line)
                    insert_class_docstring = True
                    i += 1
                    continue
                else:
                    # Multi-line class definition
                    new_lines.append(line)
                    just_saw_class = True
                    in_multiline_def = True
                    paren_depth = stripped.count('(') - stripped.count(')')
                    i += 1
                    continue
            else:
                # Fallback for other class patterns
                cm = re.match(r'^(\s*)class\s+(\w+)', line)
                if cm:
                    class_indent = len(cm.group(1))
                    class_name = cm.group(2)
                    class_bases = []
                    in_class = True

                    if ':' in stripped and not stripped.endswith('('):
                        comment = f"定义类{class_name}"
                        if not has_inline_comment(line):
                            new_lines.append(f"{line}  # {comment}")
                        else:
                            new_lines.append(line)
                        insert_class_docstring = True
                    else:
                        new_lines.append(line)
                        just_saw_class = True
                        in_multiline_def = True
                        paren_depth = stripped.count('(') - stripped.count(')')
                    i += 1
                    continue

        # Check for function/method definition
        func_match = re.match(r'^(\s*)(async\s+)?def\s+(\w+)\s*\(', line)
        if func_match:
            indent_str = func_match.group(1)
            is_async = bool(func_match.group(2))
            fname = func_match.group(3)
            is_method = len(indent_str) > 0 and in_class

            func_is_async = is_async
            func_is_method = is_method
            func_name = fname

            kind = "异步" if is_async else ""
            fkind = "方法" if is_method else "函数"
            comment = f"定义{kind}{fkind}{fname}"

            if ':' in stripped.split(')')[-1] if ')' in stripped else '':
                # Complete definition on one line
                if not has_inline_comment(line):
                    new_lines.append(f"{line}  # {comment}")
                elif not has_chinese(line):
                    new_lines.append(f"{line}  # {comment}")
                else:
                    new_lines.append(line)
                insert_func_docstring = True
            elif stripped.endswith(':'):
                if not has_inline_comment(line):
                    new_lines.append(f"{line}  # {comment}")
                elif not has_chinese(line):
                    new_lines.append(f"{line}  # {comment}")
                else:
                    new_lines.append(line)
                insert_func_docstring = True
            else:
                # Multi-line function definition
                new_lines.append(line)
                just_saw_func = True
                in_multiline_def = True
                paren_depth = stripped.count('(') - stripped.count(')')
            i += 1
            continue

        # Check for import statements
        if stripped.startswith(('import ', 'from ')):
            if '(' in stripped and ')' not in stripped:
                in_multiline_import = True
                comment = get_import_comment(line)
                if comment and not has_inline_comment(line):
                    new_lines.append(f"{line}  # {comment}")
                else:
                    new_lines.append(line)
                i += 1
                continue

            comment = get_import_comment(line)
            if comment:
                if has_inline_comment(line):
                    existing_comment = line.split('#', 1)[1]
                    if not has_chinese(existing_comment):
                        new_lines.append(f"{line}  # {comment}")
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(f"{line}  # {comment}")
            else:
                new_lines.append(line)
            i += 1
            continue

        # Check for known code patterns
        if not should_skip_line(line):
            comment = get_code_comment(line)
            if comment and not has_inline_comment(line):
                new_lines.append(f"{line}  # {comment}")
            else:
                # For code lines without a specific pattern, add generic comment
                # But skip lines that are just arguments, continuations, etc.
                if not has_inline_comment(line) and not stripped.startswith((')', ']', '}', '#', '"""', "'''", '"', "'")):
                    # Try to add meaningful comments for assignment operations
                    assign_match = re.match(r'^(\s*)(\w+)\s*=\s*(.+)', line)
                    if assign_match and not stripped.startswith('if ') and not stripped.startswith('elif '):
                        var_name = assign_match.group(2)
                        value = assign_match.group(3).strip()
                        if var_name not in ('_', '__all__'):
                            comment = f"设置变量{var_name}"
                            new_lines.append(f"{line}  # {comment}")
                        else:
                            new_lines.append(line)
                    elif re.match(r'^\s*self\.\w+\s*=', line):
                        attr_match = re.match(r'^\s*self\.(\w+)\s*=', line)
                        if attr_match:
                            comment = f"设置实例属性{attr_match.group(1)}"
                            new_lines.append(f"{line}  # {comment}")
                        else:
                            new_lines.append(line)
                    elif re.match(r'^\s*\w+\.\w+\s*=', line):
                        new_lines.append(line)
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
        else:
            new_lines.append(line)

        i += 1

    result = '\n'.join(new_lines)

    # Only write if changes were made
    if result != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(result)
        return True
    return False


def main():
    base_dir = r'E:\code\datacanvas\vllm\vllm\entrypoints'
    processed = 0
    skipped = 0
    modified = 0

    for root, dirs, files in os.walk(base_dir):
        for filename in sorted(files):
            if not filename.endswith('.py'):
                continue
            filepath = os.path.join(root, filename)
            processed += 1
            try:
                result = annotate_file(filepath)
                if result:
                    modified += 1
                    print(f"  Modified: {filepath}")
                else:
                    skipped += 1
                    print(f"  Skipped: {filepath}")
            except Exception as e:
                print(f"  ERROR on {filepath}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\nDone! Processed: {processed}, Modified: {modified}, Skipped: {skipped}")


if __name__ == '__main__':
    main()
