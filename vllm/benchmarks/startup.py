# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源协议标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明：vLLM项目贡献者
"""Benchmark the cold and warm startup time of vLLM models.
基准测试vLLM模型的冷启动和热启动时间。

This script measures total startup time (including model loading, compilation,
and cache operations) for both cold and warm scenarios:
本脚本测量冷启动和热启动场景下的总启动时间（包括模型加载、编译和缓存操作）：
- Cold startup: Fresh start with no caches (temporary cache directories)
  冷启动：无缓存的全新启动（使用临时缓存目录）
- Warm startup: Using cached compilation and model info
  热启动：使用已缓存的编译结果和模型信息
"""

import argparse  # 导入命令行参数解析模块
import dataclasses  # 导入数据类模块
import json  # 导入JSON处理模块
import multiprocessing  # 导入多进程模块
import os  # 导入操作系统接口模块
import shutil  # 导入文件操作工具模块
import tempfile  # 导入临时文件模块
import time  # 导入时间模块
from contextlib import contextmanager  # 导入上下文管理器装饰器
from typing import Any  # 导入类型标注工具

import numpy as np  # 导入NumPy数值计算库
from tqdm import tqdm  # 导入进度条库

from vllm.benchmarks.lib.utils import (  # 导入基准测试工具函数
    convert_to_pytorch_benchmark_format,  # PyTorch基准格式转换
    write_to_json,  # JSON写入工具
)
from vllm.engine.arg_utils import EngineArgs  # 导入引擎参数工具类


@contextmanager
def cold_startup():
    """
    Context manager to measure cold startup time:
    冷启动时间测量的上下文管理器：
    1. Uses a temporary directory for vLLM cache to avoid any pollution
       between cold startup iterations.
       使用临时目录作为vLLM缓存，避免冷启动迭代间的污染。
    2. Uses inductor's fresh_cache to clear torch.compile caches.
       使用inductor的fresh_cache清除torch.compile缓存。
    """
    from torch._inductor.utils import fresh_cache  # 导入torch编译缓存清理工具

    # Use temporary directory for caching to avoid any pollution between cold startups
    original_cache_root = os.environ.get("VLLM_CACHE_ROOT")  # 保存原始缓存根目录
    temp_cache_dir = tempfile.mkdtemp(prefix="vllm_startup_bench_cold_")  # 创建临时缓存目录
    try:  # 尝试执行
        os.environ["VLLM_CACHE_ROOT"] = temp_cache_dir  # 设置临时缓存目录
        with fresh_cache():  # 使用新鲜缓存上下文
            yield  # 让出控制权
    finally:  # 最终清理
        # Clean up temporary cache directory
        shutil.rmtree(temp_cache_dir, ignore_errors=True)  # 删除临时缓存目录
        if original_cache_root:  # 如果有原始缓存目录
            os.environ["VLLM_CACHE_ROOT"] = original_cache_root  # 恢复原始设置
        else:  # 如果没有原始缓存目录
            os.environ.pop("VLLM_CACHE_ROOT", None)  # 移除环境变量


def run_startup_in_subprocess(engine_args, result_queue):
    """
    Run LLM startup in a subprocess and return timing metrics via a queue.
    This ensures complete isolation between iterations.
    在子进程中运行LLM启动，并通过队列返回计时指标。
    这确保了迭代之间的完全隔离。
    """
    try:  # 尝试启动
        # Import inside the subprocess to avoid issues with forking
        from vllm import LLM  # 在子进程内导入，避免fork问题

        # Measure total startup time
        start_time = time.perf_counter()  # 记录启动开始时间

        llm = LLM(**dataclasses.asdict(engine_args))  # 创建LLM实例

        total_startup_time = time.perf_counter() - start_time  # 计算总启动时间

        # Extract compilation time if available
        compilation_time = 0.0  # 初始化编译时间
        if hasattr(llm.llm_engine, "vllm_config"):  # 如果有vllm配置
            vllm_config = llm.llm_engine.vllm_config  # 获取配置
            if (  # 如果有编译配置
                hasattr(vllm_config, "compilation_config")  # 检查编译配置属性
                and vllm_config.compilation_config is not None  # 且不为空
            ):
                compilation_time = vllm_config.compilation_config.compilation_time  # 获取编译时间

        result_queue.put(  # 将结果放入队列
            {
                "total_startup_time": total_startup_time,  # 总启动时间
                "compilation_time": compilation_time,  # 编译时间
            }
        )

    except Exception as e:  # 捕获异常
        result_queue.put(None)  # 放入None表示失败
        result_queue.put(str(e))  # 放入错误信息


def save_to_pytorch_benchmark_format(
    args: argparse.Namespace, results: dict[str, Any]  # 命令行参数和结果
) -> None:  # 无返回值
    """
    将启动时间基准测试结果保存为PyTorch基准测试格式。
    """
    base_name = os.path.splitext(args.output_json)[0]  # 获取输出文件基础名

    cold_startup_records = convert_to_pytorch_benchmark_format(  # 转换冷启动记录
        args=args,  # 参数
        metrics={  # 指标
            "avg_cold_startup_time": [results["avg_cold_startup_time"]],  # 平均冷启动时间
        },
        extra_info={  # 额外信息
            "cold_startup_times": results["cold_startup_times"],  # 冷启动时间列表
            "cold_startup_percentiles": results["cold_startup_percentiles"],  # 冷启动百分位数
        },
    )
    if cold_startup_records:  # 如果有记录
        write_to_json(f"{base_name}.cold_startup.pytorch.json", cold_startup_records)  # 写入文件

    cold_compilation_records = convert_to_pytorch_benchmark_format(  # 转换冷编译记录
        args=args,  # 参数
        metrics={  # 指标
            "avg_cold_compilation_time": [results["avg_cold_compilation_time"]],  # 平均冷编译时间
        },
        extra_info={  # 额外信息
            "cold_compilation_times": results["cold_compilation_times"],  # 冷编译时间列表
            "cold_compilation_percentiles": results["cold_compilation_percentiles"],  # 冷编译百分位数
        },
    )
    if cold_compilation_records:  # 如果有记录
        write_to_json(  # 写入文件
            f"{base_name}.cold_compilation.pytorch.json", cold_compilation_records  # 冷编译记录文件名
        )

    warm_startup_records = convert_to_pytorch_benchmark_format(  # 转换热启动记录
        args=args,  # 参数
        metrics={  # 指标
            "avg_warm_startup_time": [results["avg_warm_startup_time"]],  # 平均热启动时间
        },
        extra_info={  # 额外信息
            "warm_startup_times": results["warm_startup_times"],  # 热启动时间列表
            "warm_startup_percentiles": results["warm_startup_percentiles"],  # 热启动百分位数
        },
    )
    if warm_startup_records:  # 如果有记录
        write_to_json(f"{base_name}.warm_startup.pytorch.json", warm_startup_records)  # 写入文件

    warm_compilation_records = convert_to_pytorch_benchmark_format(  # 转换热编译记录
        args=args,  # 参数
        metrics={  # 指标
            "avg_warm_compilation_time": [results["avg_warm_compilation_time"]],  # 平均热编译时间
        },
        extra_info={  # 额外信息
            "warm_compilation_times": results["warm_compilation_times"],  # 热编译时间列表
            "warm_compilation_percentiles": results["warm_compilation_percentiles"],  # 热编译百分位数
        },
    )
    if warm_compilation_records:  # 如果有记录
        write_to_json(  # 写入文件
            f"{base_name}.warm_compilation.pytorch.json", warm_compilation_records  # 热编译记录文件名
        )


def add_cli_args(parser: argparse.ArgumentParser):
    """
    添加启动时间基准测试的命令行参数。
    """
    parser.add_argument(  # 冷启动迭代次数
        "--num-iters-cold",  # 参数名
        type=int,  # 整数类型
        default=3,  # 默认3次
        help="Number of cold startup iterations.",  # 帮助信息
    )
    parser.add_argument(  # 预热迭代次数
        "--num-iters-warmup",  # 参数名
        type=int,  # 整数类型
        default=1,  # 默认1次
        help="Number of warmup iterations before benchmarking warm startups.",  # 帮助信息
    )
    parser.add_argument(  # 热启动迭代次数
        "--num-iters-warm",  # 参数名
        type=int,  # 整数类型
        default=3,  # 默认3次
        help="Number of warm startup iterations.",  # 帮助信息
    )
    parser.add_argument(  # JSON输出路径
        "--output-json",  # 参数名
        type=str,  # 字符串类型
        default=None,  # 默认无
        help="Path to save the startup time results in JSON format.",  # 帮助信息
    )

    parser = EngineArgs.add_cli_args(parser)  # 添加引擎参数
    return parser  # 返回解析器


def main(args: argparse.Namespace):
    """
    启动时间基准测试的主函数。
    """
    # Set multiprocessing start method to 'spawn' for clean process isolation
    # This ensures each subprocess starts fresh without inheriting state
    multiprocessing.set_start_method("spawn", force=True)  # 设置多进程启动方法为spawn

    engine_args = EngineArgs.from_cli_args(args)  # 从命令行参数创建引擎参数

    def create_llm_and_measure_startup():
        """
        Create LLM instance in a subprocess and measure startup time.
        Returns timing metrics, using subprocess for complete isolation.
        在子进程中创建LLM实例并测量启动时间。
        使用子进程确保完全隔离，返回计时指标。
        """

        # Create a queue for inter-process communication
        result_queue = multiprocessing.Queue()  # 创建进程间通信队列
        process = multiprocessing.Process(  # 创建子进程
            target=run_startup_in_subprocess,  # 目标函数
            args=(  # 参数
                engine_args,  # 引擎参数
                result_queue,  # 结果队列
            ),
        )
        process.start()  # 启动子进程
        process.join()  # 等待子进程完成

        if not result_queue.empty():  # 如果队列非空
            result = result_queue.get()  # 获取结果
            if result is None:  # 如果结果为None（表示失败）
                if not result_queue.empty():  # 如果还有错误信息
                    error_msg = result_queue.get()  # 获取错误信息
                    raise RuntimeError(f"Subprocess failed: {error_msg}")  # 抛出运行时错误
                else:  # 没有错误信息
                    raise RuntimeError("Subprocess failed with unknown error")  # 抛出未知错误
            return result  # 返回结果
        else:  # 如果队列为空
            raise RuntimeError("Subprocess did not return a result")  # 抛出无结果错误

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # 禁用V1多进程以收集启动指标
    print("Setting VLLM_ENABLE_V1_MULTIPROCESSING=0 to collect startup metrics.\n")  # 打印设置信息

    print("Measuring cold startup time...\n")  # 打印冷启动测量信息
    cold_startup_times = []  # 初始化冷启动时间列表
    cold_compilation_times = []  # 初始化冷编译时间列表
    for i in tqdm(range(args.num_iters_cold), desc="Cold startup iterations"):  # 冷启动迭代
        with cold_startup():  # 使用冷启动上下文
            metrics = create_llm_and_measure_startup()  # 创建LLM并测量启动时间
            cold_startup_times.append(metrics["total_startup_time"])  # 记录启动时间
            cold_compilation_times.append(metrics["compilation_time"])  # 记录编译时间

    # Warmup for warm startup
    print("\nWarming up for warm startup measurement...\n")  # 打印热启动预热信息
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):  # 预热迭代
        create_llm_and_measure_startup()  # 执行预热

    print("\nMeasuring warm startup time...\n")  # 打印热启动测量信息
    warm_startup_times = []  # 初始化热启动时间列表
    warm_compilation_times = []  # 初始化热编译时间列表
    for i in tqdm(range(args.num_iters_warm), desc="Warm startup iterations"):  # 热启动迭代
        metrics = create_llm_and_measure_startup()  # 创建LLM并测量启动时间
        warm_startup_times.append(metrics["total_startup_time"])  # 记录启动时间
        warm_compilation_times.append(metrics["compilation_time"])  # 记录编译时间

    # Calculate statistics
    cold_startup_array = np.array(cold_startup_times)  # 转换为NumPy数组
    cold_compilation_array = np.array(cold_compilation_times)  # 转换为NumPy数组
    warm_startup_array = np.array(warm_startup_times)  # 转换为NumPy数组
    warm_compilation_array = np.array(warm_compilation_times)  # 转换为NumPy数组

    avg_cold_startup = np.mean(cold_startup_array)  # 计算平均冷启动时间
    avg_cold_compilation = np.mean(cold_compilation_array)  # 计算平均冷编译时间
    avg_warm_startup = np.mean(warm_startup_array)  # 计算平均热启动时间
    avg_warm_compilation = np.mean(warm_compilation_array)  # 计算平均热编译时间

    percentages = [10, 25, 50, 75, 90, 99]  # 百分位数列表
    cold_startup_percentiles = np.percentile(cold_startup_array, percentages)  # 冷启动百分位数
    cold_compilation_percentiles = np.percentile(cold_compilation_array, percentages)  # 冷编译百分位数
    warm_startup_percentiles = np.percentile(warm_startup_array, percentages)  # 热启动百分位数
    warm_compilation_percentiles = np.percentile(warm_compilation_array, percentages)  # 热编译百分位数

    print("\n" + "=" * 60)  # 打印分隔线
    print("STARTUP TIME BENCHMARK RESULTS")  # 打印结果标题
    print("=" * 60)  # 打印分隔线

    # Cold startup statistics
    print("\nCOLD STARTUP:")  # 打印冷启动标题
    print(f"Avg total startup time: {avg_cold_startup:.2f} seconds")  # 打印平均冷启动时间
    print(f"Avg compilation time:   {avg_cold_compilation:.2f} seconds")  # 打印平均冷编译时间
    print("Startup time percentiles:")  # 打印启动时间百分位数
    for percentage, percentile in zip(percentages, cold_startup_percentiles):  # 遍历百分位数
        print(f"  {percentage}%: {percentile:.2f} seconds")  # 打印各百分位
    print("Compilation time percentiles:")  # 打印编译时间百分位数
    for percentage, percentile in zip(percentages, cold_compilation_percentiles):  # 遍历百分位数
        print(f"  {percentage}%: {percentile:.2f} seconds")  # 打印各百分位

    # Warm startup statistics
    print("\nWARM STARTUP:")  # 打印热启动标题
    print(f"Avg total startup time: {avg_warm_startup:.2f} seconds")  # 打印平均热启动时间
    print(f"Avg compilation time:   {avg_warm_compilation:.2f} seconds")  # 打印平均热编译时间
    print("Startup time percentiles:")  # 打印启动时间百分位数
    for percentage, percentile in zip(percentages, warm_startup_percentiles):  # 遍历百分位数
        print(f"  {percentage}%: {percentile:.2f} seconds")  # 打印各百分位
    print("Compilation time percentiles:")  # 打印编译时间百分位数
    for percentage, percentile in zip(percentages, warm_compilation_percentiles):  # 遍历百分位数
        print(f"  {percentage}%: {percentile:.2f} seconds")  # 打印各百分位

    print("=" * 60)  # 打印分隔线

    # Output JSON results if specified
    if args.output_json:  # 如果指定了JSON输出路径
        results = {  # 构建结果字典
            "avg_cold_startup_time": float(avg_cold_startup),  # 平均冷启动时间
            "avg_cold_compilation_time": float(avg_cold_compilation),  # 平均冷编译时间
            "cold_startup_times": cold_startup_times,  # 冷启动时间列表
            "cold_compilation_times": cold_compilation_times,  # 冷编译时间列表
            "cold_startup_percentiles": dict(  # 冷启动百分位数字典
                zip(percentages, cold_startup_percentiles.tolist())  # 百分位数映射
            ),
            "cold_compilation_percentiles": dict(  # 冷编译百分位数字典
                zip(percentages, cold_compilation_percentiles.tolist())  # 百分位数映射
            ),
            "avg_warm_startup_time": float(avg_warm_startup),  # 平均热启动时间
            "avg_warm_compilation_time": float(avg_warm_compilation),  # 平均热编译时间
            "warm_startup_times": warm_startup_times,  # 热启动时间列表
            "warm_compilation_times": warm_compilation_times,  # 热编译时间列表
            "warm_startup_percentiles": dict(  # 热启动百分位数字典
                zip(percentages, warm_startup_percentiles.tolist())  # 百分位数映射
            ),
            "warm_compilation_percentiles": dict(  # 热编译百分位数字典
                zip(percentages, warm_compilation_percentiles.tolist())  # 百分位数映射
            ),
        }
        with open(args.output_json, "w") as f:  # 打开文件写入
            json.dump(results, f, indent=4)  # 序列化为JSON
        save_to_pytorch_benchmark_format(args, results)  # 保存为PyTorch格式
