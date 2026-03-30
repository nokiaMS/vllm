# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源协议标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明：vLLM项目贡献者

import argparse  # 导入命令行参数解析模块
import json  # 导入JSON处理模块
import math  # 导入数学运算模块
import os  # 导入操作系统接口模块
from contextlib import contextmanager  # 导入上下文管理器工具
from typing import Any  # 导入类型标注


def extract_field(
    args: argparse.Namespace, extra_info: dict[str, Any], field_name: str  # 从参数或额外信息中提取字段
) -> str:  # 返回字符串
    """
    从命令行参数或额外信息字典中提取指定字段的值。
    """
    if field_name in extra_info:  # 如果字段在额外信息中
        return extra_info[field_name]  # 直接返回额外信息中的值

    v = args  # 从参数对象开始
    # For example, args.compilation_config.mode
    for nested_field in field_name.split("."):  # 按点号分割支持嵌套字段
        if not hasattr(v, nested_field):  # 如果属性不存在
            return ""  # 返回空字符串
        v = getattr(v, nested_field)  # 获取嵌套属性值
    return v  # 返回找到的值


def use_compile(args: argparse.Namespace, extra_info: dict[str, Any]) -> bool:
    """
    Check if the benchmark is run with torch.compile
    检查基准测试是否使用了torch.compile编译。
    """
    return not (  # 返回是否使用编译的布尔值
        extract_field(args, extra_info, "compilation_config.mode") == "0"  # 编译模式不是"0"
        or "eager" in getattr(args, "output_json", "")  # 输出文件名中不包含"eager"
        or "eager" in getattr(args, "result_filename", "")  # 结果文件名中不包含"eager"
    )


def convert_to_pytorch_benchmark_format(
    args: argparse.Namespace, metrics: dict[str, list], extra_info: dict[str, Any]  # 转换为PyTorch基准测试格式
) -> list:  # 返回记录列表
    """
    Save the benchmark results in the format used by PyTorch OSS benchmark with
    on metric per record
    将基准测试结果保存为PyTorch OSS基准测试使用的格式，每个指标一条记录。
    https://github.com/pytorch/pytorch/wiki/How-to-integrate-with-PyTorch-OSS-benchmark-database
    """
    records = []  # 初始化记录列表
    if not os.environ.get("SAVE_TO_PYTORCH_BENCHMARK_FORMAT", False):  # 检查环境变量是否启用
        return records  # 如果未启用则返回空列表

    for name, benchmark_values in metrics.items():  # 遍历所有指标
        if not isinstance(benchmark_values, list):  # 验证指标值必须是列表
            raise TypeError(  # 抛出类型错误
                f"benchmark_values for metric '{name}' must be a list, "  # 错误消息
                f"but got {type(benchmark_values).__name__}"  # 显示实际类型
            )

        record = {  # 构建单条记录
            "benchmark": {  # 基准测试信息
                "name": "vLLM benchmark",  # 基准测试名称
                "extra_info": {  # 额外信息
                    "args": vars(args),  # 命令行参数
                    "compilation_config.mode": extract_field(  # 编译配置模式
                        args, extra_info, "compilation_config.mode"  # 提取编译配置
                    ),
                    "optimization_level": extract_field(  # 优化级别
                        args, extra_info, "optimization_level"  # 提取优化级别
                    ),
                    # A boolean field used by vLLM benchmark HUD dashboard
                    "use_compile": use_compile(args, extra_info),  # 是否使用编译
                },
            },
            "model": {  # 模型信息
                "name": args.model,  # 模型名称
            },
            "metric": {  # 指标信息
                "name": name,  # 指标名称
                "benchmark_values": benchmark_values,  # 基准测试值
                "extra_info": extra_info,  # 额外信息
            },
        }

        tp = record["benchmark"]["extra_info"]["args"].get("tensor_parallel_size")  # 获取张量并行大小
        # Save tensor_parallel_size parameter if it's part of the metadata
        if not tp and "tensor_parallel_size" in extra_info:  # 如果参数中没有但额外信息中有
            record["benchmark"]["extra_info"]["args"]["tensor_parallel_size"] = (  # 保存张量并行大小
                extra_info["tensor_parallel_size"]  # 从额外信息中获取
            )

        records.append(record)  # 将记录添加到列表

    return records  # 返回所有记录


class InfEncoder(json.JSONEncoder):
    """
    自定义JSON编码器，处理无穷大和非JSON可序列化类型。
    """
    def clear_inf(self, o: Any):  # 清理无穷大值
        """
        递归替换数据中的无穷大浮点数为字符串"inf"。
        """
        if isinstance(o, dict):  # 如果是字典
            return {  # 递归处理字典
                str(k)  # 将非标准键类型转换为字符串
                if not isinstance(k, (str, int, float, bool, type(None)))  # 检查键类型
                else k: self.clear_inf(v)  # 递归处理值
                for k, v in o.items()  # 遍历字典项
            }
        elif isinstance(o, list):  # 如果是列表
            return [self.clear_inf(v) for v in o]  # 递归处理列表元素
        elif isinstance(o, float) and math.isinf(o):  # 如果是无穷大浮点数
            return "inf"  # 替换为字符串"inf"
        return o  # 其他类型直接返回

    def iterencode(self, o: Any, *args, **kwargs) -> Any:
        """
        重写编码方法，在编码前清理无穷大值。
        """
        return super().iterencode(self.clear_inf(o), *args, **kwargs)  # 清理后调用父类编码


def write_to_json(filename: str, records: list) -> None:
    """
    将记录列表写入JSON文件。
    """
    with open(filename, "w") as f:  # 打开文件进行写入
        json.dump(  # 序列化为JSON
            records,  # 写入的记录
            f,  # 文件对象
            cls=InfEncoder,  # 使用自定义编码器
            default=lambda o: f"<{type(o).__name__} is not JSON serializable>",  # 处理不可序列化对象
        )


@contextmanager
def default_vllm_config():
    """Set a default VllmConfig for cases that directly test CustomOps or pathways
    that use get_current_vllm_config() outside of a full engine context.
    为直接测试CustomOps或在完整引擎上下文之外使用get_current_vllm_config()的情况设置默认VllmConfig。
    """
    from vllm.config import VllmConfig, set_current_vllm_config  # 导入vLLM配置相关模块

    with set_current_vllm_config(VllmConfig()):  # 设置默认配置
        yield  # 让出控制权
