# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源协议标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明：vLLM项目贡献者
import json  # 导入JSON处理模块
import os  # 导入操作系统接口模块
from typing import Any  # 导入类型标注工具


class ParameterSweep(list["ParameterSweepItem"]):
    """
    参数扫描列表类，继承自list，用于管理多组参数组合。
    """
    @classmethod
    def read_json(cls, filepath: os.PathLike):
        """
        从JSON文件读取参数扫描配置。
        """
        with open(filepath, "rb") as f:  # 以二进制模式打开文件
            data = json.load(f)  # 加载JSON数据

        # Support both list and dict formats
        if isinstance(data, dict):  # 如果是字典格式
            return cls.read_from_dict(data)  # 从字典格式读取

        return cls.from_records(data)  # 从记录列表格式读取

    @classmethod
    def read_from_dict(cls, data: dict[str, dict[str, object]]):
        """
        Read parameter sweep from a dict format where keys are names.
        从字典格式读取参数扫描，字典的键作为名称。

        Example:
            {
                "experiment1": {"max_tokens": 100, "temperature": 0.7},
                "experiment2": {"max_tokens": 200, "temperature": 0.9}
            }
        """
        records = [{"_benchmark_name": name, **params} for name, params in data.items()]  # 将字典转换为记录列表
        return cls.from_records(records)  # 从记录创建实例

    @classmethod
    def from_records(cls, records: list[dict[str, object]]):
        """
        从记录列表创建参数扫描实例，验证格式和唯一性。
        """
        if not isinstance(records, list):  # 验证必须是列表
            raise TypeError(  # 抛出类型错误
                f"The parameter sweep should be a list of dictionaries, "  # 参数扫描应为字典列表
                f"but found type: {type(records)}"  # 实际类型
            )

        # Validate that all _benchmark_name values are unique if provided
        names = [r["_benchmark_name"] for r in records if "_benchmark_name" in r]  # 提取所有名称
        if names and len(names) != len(set(names)):  # 检查名称唯一性
            duplicates = [name for name in names if names.count(name) > 1]  # 找出重复名称
            raise ValueError(  # 抛出值错误
                f"Duplicate _benchmark_name values found: {set(duplicates)}. "  # 发现重复名称
                f"All _benchmark_name values must be unique."  # 所有名称必须唯一
            )

        return cls(ParameterSweepItem.from_record(record) for record in records)  # 创建实例


class ParameterSweepItem(dict[str, object]):
    """
    参数扫描项类，继承自dict，表示单组参数组合。
    """
    @classmethod
    def from_record(cls, record: dict[str, object]):
        """
        从单条记录创建参数扫描项。
        """
        if not isinstance(record, dict):  # 验证必须是字典
            raise TypeError(  # 抛出类型错误
                f"Each item in the parameter sweep should be a dictionary, "  # 每项应为字典
                f"but found type: {type(record)}"  # 实际类型
            )

        return cls(record)  # 创建实例

    def __or__(self, other: dict[str, Any]):
        """
        重写合并运算符，保持返回类型为ParameterSweepItem。
        """
        return type(self)(super().__or__(other))  # 合并字典并保持类型

    @property
    def name(self) -> str:
        """
        Get the name for this parameter sweep item.
        获取此参数扫描项的名称。

        Returns the '_benchmark_name' field if present, otherwise returns a text
        representation of all parameters.
        如果存在'_benchmark_name'字段则返回该值，否则返回所有参数的文本表示。
        """
        if "_benchmark_name" in self:  # 如果有基准测试名称
            return str(self["_benchmark_name"])  # 返回名称

        return self.as_text(sep="-")  # 否则返回参数文本表示

    # In JSON, we prefer "_"
    def _iter_param_key_candidates(self, param_key: str):
        """
        生成参数键的各种变体候选（下划线和连字符互换）。
        """
        # Inner config arguments are not converted by the CLI
        if "." in param_key:  # 如果键包含点号（嵌套参数）
            prefix, rest = param_key.split(".", 1)  # 分割前缀和剩余部分
            for prefix_candidate in self._iter_param_key_candidates(prefix):  # 递归处理前缀
                yield prefix_candidate + "." + rest  # 生成候选键

            return  # 返回

        yield param_key  # 生成原始键
        yield param_key.replace("-", "_")  # 生成下划线版本
        yield param_key.replace("_", "-")  # 生成连字符版本

    # In CLI, we prefer "-"
    def _iter_cmd_key_candidates(self, param_key: str):
        """
        生成命令行参数键的候选形式（带--前缀）。
        """
        for k in reversed(tuple(self._iter_param_key_candidates(param_key))):  # 反序遍历参数键候选
            yield "--" + k  # 生成带前缀的命令行参数

    def _normalize_cmd_key(self, param_key: str):
        """
        将参数键标准化为命令行格式。
        """
        return next(self._iter_cmd_key_candidates(param_key))  # 返回第一个候选

    def has_param(self, param_key: str) -> bool:
        """
        检查是否包含指定的参数键（考虑各种变体形式）。
        """
        return any(k in self for k in self._iter_param_key_candidates(param_key))  # 检查任意变体是否存在

    def _normalize_cmd_kv_pair(self, k: str, v: object) -> list[str]:
        """
        Normalize a key-value pair into command-line arguments.
        将键值对标准化为命令行参数。

        Returns a list containing either:
        - A single element for boolean flags (e.g., ['--flag'] or ['--flag=true'])
          布尔标志返回单个元素
        - Two elements for key-value pairs (e.g., ['--key', 'value'])
          键值对返回两个元素
        """
        if isinstance(v, bool):  # 如果是布尔值
            # For nested params (containing "."), use =true/false syntax
            if "." in k:  # 嵌套参数使用=true/false语法
                return [f"{self._normalize_cmd_key(k)}={'true' if v else 'false'}"]  # 生成嵌套布尔参数
            else:  # 普通布尔参数
                return [self._normalize_cmd_key(k if v else "no-" + k)]  # 真值用原键，假值加no-前缀
        else:  # 非布尔值
            return [self._normalize_cmd_key(k), str(v)]  # 返回键值对

    def apply_to_cmd(self, cmd: list[str]) -> list[str]:
        """
        将参数应用到命令行列表中，替换已有参数或添加新参数。
        """
        cmd = list(cmd)  # 复制命令列表

        for k, v in self.items():  # 遍历所有参数
            # Skip the '_benchmark_name' field, not a parameter
            if k == "_benchmark_name":  # 跳过基准测试名称字段
                continue  # 继续下一个参数

            # Serialize dict values as JSON
            if isinstance(v, dict):  # 如果值是字典
                v = json.dumps(v)  # 序列化为JSON字符串

            for k_candidate in self._iter_cmd_key_candidates(k):  # 遍历键的候选形式
                try:  # 尝试查找现有参数
                    k_idx = cmd.index(k_candidate)  # 查找参数在命令中的位置

                    # Replace existing parameter
                    normalized = self._normalize_cmd_kv_pair(k, v)  # 标准化键值对
                    if len(normalized) == 1:  # 如果是布尔标志
                        # Boolean flag
                        cmd[k_idx] = normalized[0]  # 替换标志
                    else:  # 如果是键值对
                        # Key-value pair
                        cmd[k_idx] = normalized[0]  # 替换键
                        cmd[k_idx + 1] = normalized[1]  # 替换值

                    break  # 找到并替换后退出
                except ValueError:  # 如果未找到
                    continue  # 尝试下一个候选
            else:  # 如果所有候选都未找到
                # Add new parameter
                cmd.extend(self._normalize_cmd_kv_pair(k, v))  # 添加新参数

        return cmd  # 返回修改后的命令

    def as_text(self, sep: str = ", ") -> str:
        """
        将参数转换为可读的文本表示，排除基准测试名称。
        """
        return sep.join(f"{k}={v}" for k, v in self.items() if k != "_benchmark_name")  # 格式化为key=value
