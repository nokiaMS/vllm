#!/usr/bin/env python3
"""
Script to add Chinese comments to Python files in specified directories.
Rules:
1. Add Chinese inline comments to each code line
2. Add Chinese docstrings to classes and functions
3. Skip lines that already have Chinese comments
4. Don't modify original content
5. Skip blank lines and pure comment lines
6. Add brief Chinese descriptions to import statements
"""

import os
import re
import ast
import sys
import textwrap


def has_chinese(text):
    """Check if text contains Chinese characters."""
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def is_blank_or_comment(line):
    """Check if a line is blank or a pure comment line."""
    stripped = line.strip()
    return stripped == '' or stripped.startswith('#')


def is_decorator_line(line):
    """Check if a line is a decorator."""
    return line.strip().startswith('@')


def is_string_only_line(line):
    """Check if line is only a string literal (part of docstring)."""
    stripped = line.strip()
    return (stripped.startswith('"""') or stripped.startswith("'''") or
            stripped.startswith('"') or stripped.startswith("'"))


def get_import_comment(line):
    """Generate a Chinese comment for an import statement."""
    stripped = line.strip()

    # Common module mappings
    import_comments = {
        'torch': '导入PyTorch深度学习框架',
        'os': '导入操作系统接口模块',
        'sys': '导入系统相关模块',
        're': '导入正则表达式模块',
        'json': '导入JSON处理模块',
        'math': '导入数学运算模块',
        'typing': '导入类型注解模块',
        'abc': '导入抽象基类模块',
        'enum': '导入枚举类型模块',
        'dataclasses': '导入数据类模块',
        'functools': '导入函数工具模块',
        'collections': '导入集合数据类型模块',
        'collections.abc': '导入集合抽象基类模块',
        'contextlib': '导入上下文管理工具模块',
        'itertools': '导入迭代工具模块',
        'importlib': '导入动态导入模块',
        'logging': '导入日志模块',
        'pathlib': '导入路径操作模块',
        'copy': '导入对象复制模块',
        'time': '导入时间处理模块',
        'threading': '导入线程模块',
        'asyncio': '导入异步IO模块',
        'numpy': '导入NumPy数值计算库',
        'regex': '导入增强正则表达式模块',
    }

    if stripped.startswith('from __future__'):
        return '从future模块导入特性'

    if stripped.startswith('import ') and not stripped.startswith('from'):
        # import X [as Y]
        parts = stripped[7:].split(' as ')[0].split(',')[0].strip()
        mod = parts.split('.')[0]
        if mod in import_comments:
            return import_comments[mod]
        return f'导入{mod}模块'

    if stripped.startswith('from '):
        # from X import Y
        match = re.match(r'from\s+([\w.]+)\s+import', stripped)
        if match:
            mod_path = match.group(1)
            mod = mod_path.split('.')[0]

            # Handle relative imports
            if mod_path.startswith('.'):
                return '从相对路径导入模块组件'

            if mod_path in import_comments:
                return import_comments[mod_path]
            if mod == 'vllm':
                # Describe vllm submodule imports
                if 'quantization' in mod_path:
                    return '从vLLM量化模块导入组件'
                if 'layers' in mod_path:
                    return '从vLLM模型层模块导入组件'
                if 'model_executor' in mod_path:
                    return '从vLLM模型执行器导入组件'
                if 'distributed' in mod_path:
                    return '从vLLM分布式模块导入组件'
                if 'kv_transfer' in mod_path:
                    return '从vLLM KV缓存传输模块导入组件'
                if 'logger' in mod_path:
                    return '从vLLM日志模块导入组件'
                if 'config' in mod_path:
                    return '从vLLM配置模块导入组件'
                if 'platforms' in mod_path:
                    return '从vLLM平台模块导入组件'
                if 'attention' in mod_path:
                    return '从vLLM注意力模块导入组件'
                if 'worker' in mod_path:
                    return '从vLLM工作进程模块导入组件'
                if 'fused_moe' in mod_path:
                    return '从vLLM融合MoE模块导入组件'
                if 'parameter' in mod_path:
                    return '从vLLM参数模块导入组件'
                if 'utils' in mod_path:
                    return '从vLLM工具模块导入组件'
                if 'scalar_type' in mod_path:
                    return '从vLLM标量类型模块导入组件'
                if 'triton' in mod_path:
                    return '从vLLM Triton工具模块导入组件'
                if 'custom_ops' in mod_path or '_custom_ops' in mod_path:
                    return '从vLLM自定义算子模块导入组件'
                if 'outputs' in mod_path:
                    return '从vLLM输出模块导入组件'
                return '从vLLM框架导入组件'

            if mod == 'compressed_tensors':
                return '从compressed_tensors压缩张量库导入组件'
            if mod == 'torch':
                return '从PyTorch框架导入组件'
            if mod_path in import_comments:
                return import_comments[mod_path]
            if mod in import_comments:
                return import_comments[mod]
            return f'从{mod}模块导入组件'

    return '导入模块'


def get_code_comment(line, context=None):
    """Generate a Chinese comment for a code line based on its content."""
    stripped = line.strip()

    # Skip lines that end with continuation
    if stripped.endswith('\\'):
        return None
    # Skip lines that are just closing brackets/parens
    if stripped in (')', ']', '}', '),', '],', '},', '):', ');'):
        return None
    # Skip very short fragments
    if len(stripped) <= 2:
        return None

    # Common patterns
    if stripped.startswith('return '):
        val = stripped[7:].strip()
        if val == 'None':
            return '返回None'
        if val == 'True':
            return '返回True'
        if val == 'False':
            return '返回False'
        if val == 'self':
            return '返回自身实例'
        return '返回结果'
    if stripped == 'return':
        return '返回'
    if stripped.startswith('raise '):
        if 'NotImplementedError' in stripped:
            return '抛出未实现错误'
        if 'ValueError' in stripped:
            return '抛出值错误异常'
        if 'RuntimeError' in stripped:
            return '抛出运行时错误'
        if 'AttributeError' in stripped:
            return '抛出属性错误异常'
        if 'TypeError' in stripped:
            return '抛出类型错误异常'
        return '抛出异常'
    if stripped.startswith('assert '):
        return '断言条件检查'
    if stripped.startswith('yield '):
        return '生成器产出值'
    if stripped == 'yield':
        return '生成器产出'
    if stripped.startswith('yield from '):
        return '从子生成器产出值'
    if stripped.startswith('pass'):
        return '占位语句'
    if stripped.startswith('continue'):
        return '继续下一次循环'
    if stripped.startswith('break'):
        return '跳出循环'
    if stripped.startswith('del '):
        return '删除变量或属性'
    if stripped.startswith('global '):
        return '声明全局变量'
    if stripped.startswith('nonlocal '):
        return '声明非局部变量'

    # Control flow
    if stripped.startswith('if ') or stripped == 'if':
        return '条件判断'
    if stripped.startswith('elif '):
        return '否则如果条件判断'
    if stripped == 'else:':
        return '否则分支'
    if stripped.startswith('for '):
        return '循环遍历'
    if stripped.startswith('while '):
        return '条件循环'
    if stripped.startswith('with '):
        return '上下文管理器'
    if stripped.startswith('try:'):
        return '异常捕获开始'
    if stripped.startswith('except ') or stripped == 'except:':
        return '异常处理'
    if stripped.startswith('finally:'):
        return '最终执行块'

    # Assignments and operations
    if stripped.startswith('self.') and '=' in stripped and not stripped.startswith('self.') == False:
        attr = stripped.split('=')[0].strip().replace('self.', '')
        if '(' not in attr and '[' not in attr:
            return f'设置实例属性{attr}'
    if stripped.startswith('super().__init__'):
        return '调用父类初始化方法'
    if stripped.startswith('super().'):
        return '调用父类方法'

    # Logging
    if 'logger.info' in stripped:
        return '记录信息日志'
    if 'logger.debug' in stripped:
        return '记录调试日志'
    if 'logger.warning' in stripped:
        return '记录警告日志'
    if 'logger.error' in stripped:
        return '记录错误日志'
    if 'init_logger' in stripped:
        return '初始化日志记录器'

    # Registration
    if '.register_parameter' in stripped:
        return '注册模型参数'
    if '.register_module' in stripped:
        return '注册子模块'
    if '.register_connector' in stripped:
        return '注册连接器'

    # Common operations
    if 'torch.empty(' in stripped:
        return '创建未初始化张量'
    if 'torch.zeros(' in stripped:
        return '创建全零张量'
    if 'torch.ones(' in stripped:
        return '创建全一张量'
    if 'torch.tensor(' in stripped:
        return '创建张量'
    if 'torch.nn.Parameter(' in stripped:
        return '创建可训练参数'
    if '.to(' in stripped and ('dtype' in stripped or 'device' in stripped or 'torch.' in stripped):
        return '转换数据类型或设备'
    if '.data.copy_(' in stripped:
        return '复制数据'
    if '.index_select(' in stripped:
        return '按索引选取元素'
    if '.index_copy_(' in stripped:
        return '按索引复制数据'
    if '.permute(' in stripped:
        return '张量维度重排'
    if '.reshape(' in stripped or '.view(' in stripped:
        return '张量形状变换'
    if '.flatten(' in stripped:
        return '张量展平'
    if '.unflatten(' in stripped:
        return '张量反展平'
    if '.transpose(' in stripped or '.t()' in stripped:
        return '张量转置'
    if '.contiguous()' in stripped:
        return '确保张量内存连续'

    # Class/function definitions are handled separately
    if stripped.startswith('class '):
        return None
    if stripped.startswith('def '):
        return None
    if stripped.startswith('async def '):
        return None

    # Multiline expressions - skip inner lines
    if stripped.startswith(('(', '[', '{')):
        return None
    # String concatenation continuations
    if stripped.startswith(('f"', 'f\'', '"', '\'')):
        return None

    # Variable assignments
    if '=' in stripped and not stripped.startswith(('if ', 'elif ', 'while ', 'for ', 'assert ', 'return ')):
        if not any(op in stripped for op in ['==', '!=', '<=', '>=', '+=', '-=', '*=', '/=']):
            parts = stripped.split('=', 1)
            lhs = parts[0].strip()
            # Simple variable names
            if re.match(r'^[a-zA-Z_][\w.]*$', lhs) and len(lhs) < 40:
                if lhs.startswith('self.'):
                    attr_name = lhs[5:]
                    return f'设置{attr_name}属性'
                return f'定义变量{lhs}'

    return None


def get_class_docstring(class_name, bases, body_summary=""):
    """Generate a Chinese docstring for a class."""
    base_str = ""
    if bases:
        base_str = f"继承自{'、'.join(bases)}。"

    return f'"""{class_name}类。\n\n    {base_str}\n    """'


def get_function_docstring(func_name, args, is_method=False):
    """Generate a Chinese docstring for a function/method."""
    special_methods = {
        '__init__': '初始化方法。',
        '__repr__': '返回对象的字符串表示。',
        '__str__': '返回对象的字符串形式。',
        '__len__': '返回对象的长度。',
        '__getitem__': '获取指定索引的元素。',
        '__setitem__': '设置指定索引的元素。',
        '__delitem__': '删除指定索引的元素。',
        '__contains__': '检查是否包含指定元素。',
        '__iter__': '返回迭代器。',
        '__next__': '返回下一个元素。',
        '__call__': '使对象可调用。',
        '__enter__': '进入上下文管理器。',
        '__exit__': '退出上下文管理器。',
        '__eq__': '判断相等。',
        '__ne__': '判断不等。',
        '__lt__': '判断小于。',
        '__gt__': '判断大于。',
        '__hash__': '返回哈希值。',
        '__post_init__': '数据类后初始化处理。',
    }

    if func_name in special_methods:
        return f'"""{special_methods[func_name]}"""'

    return f'"""{func_name}方法。"""'


def process_file(filepath):
    """Process a single Python file to add Chinese comments."""
    with open(filepath, 'r', encoding='utf-8') as f:
        original_lines = f.readlines()

    if not original_lines:
        return

    # Parse AST to find classes and functions
    source = ''.join(original_lines)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        print(f"  Skipping {filepath} due to syntax error")
        return

    # Collect class and function info
    class_lines = {}  # line_no -> (class_name, bases)
    func_lines = {}   # line_no -> (func_name, args, is_method)
    docstring_lines = set()  # lines that are part of existing docstrings

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    bases.append(ast.dump(base))
            class_lines[node.lineno] = (node.name, bases)

            # Check for existing docstring
            if (node.body and isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, (ast.Str, ast.Constant))):
                ds_node = node.body[0]
                for dl in range(ds_node.lineno, ds_node.end_lineno + 1):
                    docstring_lines.add(dl)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.args]
            is_method = len(args) > 0 and args[0] == 'self'
            func_lines[node.lineno] = (node.name, args, is_method)

            # Check for existing docstring
            if (node.body and isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, (ast.Str, ast.Constant))):
                ds_node = node.body[0]
                for dl in range(ds_node.lineno, ds_node.end_lineno + 1):
                    docstring_lines.add(dl)

    # Track multiline string state
    result_lines = []
    in_multiline_string = False
    multiline_quote = None
    in_multiline_paren = 0  # Track nested parentheses
    skip_docstring_insert = set()  # Lines after which we already have docstrings

    # First pass: identify lines that already have docstrings after class/func defs
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, (ast.Str, ast.Constant))):
                skip_docstring_insert.add(node.lineno)

    for i, line in enumerate(original_lines):
        line_no = i + 1
        stripped = line.rstrip('\n\r')
        content = stripped.strip()

        # Handle multiline strings
        if in_multiline_string:
            result_lines.append(line)
            if multiline_quote in content:
                in_multiline_string = False
            continue

        # Check for multiline string start
        for quote in ['"""', "'''"]:
            count = content.count(quote)
            if count == 1:  # Opening quote without closing
                in_multiline_string = True
                multiline_quote = quote
                break

        # Skip if in docstring range
        if line_no in docstring_lines:
            result_lines.append(line)
            continue

        # Blank lines
        if content == '':
            result_lines.append(line)
            continue

        # Pure comment lines - skip
        if content.startswith('#'):
            result_lines.append(line)
            continue

        # Already has Chinese comment
        if has_chinese(stripped):
            result_lines.append(line)
            continue

        # Existing inline comment check
        has_inline_comment = False
        # Simple check - if there's a # not inside a string
        in_str = False
        str_char = None
        for ci, c in enumerate(stripped):
            if c in ('"', "'") and not in_str:
                in_str = True
                str_char = c
            elif c == str_char and in_str:
                in_str = False
            elif c == '#' and not in_str:
                has_inline_comment = True
                break

        # Handle import lines
        if content.startswith(('import ', 'from ')) and not content.startswith('from .'):
            comment = get_import_comment(content)
            if comment and not has_inline_comment:
                # Don't add if line is too long already
                if len(stripped) < 80:
                    result_lines.append(stripped + '  # ' + comment + '\n')
                else:
                    result_lines.append(line)
            else:
                result_lines.append(line)
            continue

        # Relative imports
        if content.startswith('from .'):
            if not has_inline_comment and len(stripped) < 80:
                result_lines.append(stripped + '  # 从相对路径导入模块组件\n')
            else:
                result_lines.append(line)
            continue

        # Class definitions - add docstring if needed
        if line_no in class_lines:
            class_name, bases = class_lines[line_no]
            result_lines.append(line)
            # Add docstring if class doesn't already have one
            if line_no not in skip_docstring_insert:
                indent = len(line) - len(line.lstrip()) + 4
                base_desc = ""
                if bases:
                    clean_bases = []
                    for b in bases:
                        if '(' in b:
                            # AST dump format, try to extract name
                            name_match = re.search(r"id='(\w+)'", b)
                            if name_match:
                                clean_bases.append(name_match.group(1))
                            else:
                                attr_match = re.search(r"attr='(\w+)'", b)
                                if attr_match:
                                    clean_bases.append(attr_match.group(1))
                        else:
                            clean_bases.append(b)
                    if clean_bases:
                        base_desc = f"，继承自{'、'.join(clean_bases)}"
                ds = f'{" " * indent}"""{class_name}类{base_desc}。"""\n'
                result_lines.append(ds)
            continue

        # Function definitions - add docstring if needed
        if line_no in func_lines:
            func_name, args, is_method = func_lines[line_no]
            result_lines.append(line)
            # Check if definition continues on next line
            if content.endswith(':') and line_no not in skip_docstring_insert:
                indent = len(line) - len(line.lstrip()) + 4
                if func_name.startswith('__') and func_name.endswith('__'):
                    special_methods = {
                        '__init__': '初始化方法',
                        '__repr__': '返回对象的字符串表示',
                        '__str__': '返回对象的字符串形式',
                        '__len__': '返回对象的长度',
                        '__call__': '使对象可调用',
                        '__enter__': '进入上下文管理器',
                        '__exit__': '退出上下文管理器',
                        '__eq__': '判断是否相等',
                        '__hash__': '返回哈希值',
                        '__post_init__': '数据类后初始化处理',
                    }
                    desc = special_methods.get(func_name, f'{func_name}特殊方法')
                else:
                    desc = f'{func_name}方法' if is_method else f'{func_name}函数'
                ds = f'{" " * indent}"""{desc}。"""\n'
                result_lines.append(ds)
            continue

        # Regular code lines - add inline comment
        if not in_multiline_string and not has_inline_comment:
            comment = get_code_comment(content)
            if comment:
                # Don't make lines too long
                if len(stripped) + len(comment) + 4 < 120:
                    result_lines.append(stripped + '  # ' + comment + '\n')
                else:
                    result_lines.append(line)
            else:
                result_lines.append(line)
        else:
            result_lines.append(line)

    # Write result
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(result_lines)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    directories = [
        os.path.join(base_dir, 'vllm', 'model_executor', 'layers', 'quantization', 'compressed_tensors'),
        os.path.join(base_dir, 'vllm', 'distributed', 'kv_transfer'),
    ]

    for dir_path in directories:
        if not os.path.exists(dir_path):
            print(f"Directory not found: {dir_path}")
            continue

        for root, dirs, files in os.walk(dir_path):
            for filename in sorted(files):
                if filename.endswith('.py'):
                    filepath = os.path.join(root, filename)
                    print(f"Processing: {filepath}")
                    try:
                        process_file(filepath)
                    except Exception as e:
                        print(f"  Error processing {filepath}: {e}")
                        import traceback
                        traceback.print_exc()


if __name__ == '__main__':
    main()
