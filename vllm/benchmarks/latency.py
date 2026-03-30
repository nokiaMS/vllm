# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源协议标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明：vLLM项目贡献者
"""Benchmark the latency of processing a single batch of requests."""  # 基准测试处理单批请求的延迟
"""基准测试处理单批请求的延迟。"""

import argparse  # 导入命令行参数解析模块
import dataclasses  # 导入数据类模块
import json  # 导入JSON处理模块
import os  # 导入操作系统接口模块
import time  # 导入时间模块
from typing import Any  # 导入类型标注工具

import numpy as np  # 导入NumPy数值计算库
from tqdm import tqdm  # 导入进度条库

from vllm.benchmarks.lib.utils import convert_to_pytorch_benchmark_format, write_to_json  # 导入PyTorch基准格式转换和JSON写入工具
from vllm.engine.arg_utils import EngineArgs  # 导入引擎参数工具类
from vllm.inputs import PromptType  # 导入提示类型
from vllm.sampling_params import BeamSearchParams  # 导入束搜索参数


def save_to_pytorch_benchmark_format(
    args: argparse.Namespace, results: dict[str, Any]  # 命令行参数和结果字典
) -> None:  # 无返回值
    """
    将延迟基准测试结果保存为PyTorch基准测试格式。
    """
    pt_records = convert_to_pytorch_benchmark_format(  # 转换为PyTorch格式
        args=args,  # 命令行参数
        metrics={"latency": results["latencies"]},  # 延迟指标
        extra_info={k: results[k] for k in ["avg_latency", "percentiles"]},  # 额外信息
    )
    if pt_records:  # 如果有记录
        pt_file = f"{os.path.splitext(args.output_json)[0]}.pytorch.json"  # 构建输出文件名
        write_to_json(pt_file, pt_records)  # 写入JSON文件


def add_cli_args(parser: argparse.ArgumentParser):
    """
    添加延迟基准测试的命令行参数。
    """
    parser.add_argument("--input-len", type=int, default=32)  # 输入长度，默认32
    parser.add_argument("--output-len", type=int, default=128)  # 输出长度，默认128
    parser.add_argument("--batch-size", type=int, default=8)  # 批次大小，默认8
    parser.add_argument(  # 每个提示生成的序列数
        "--n",  # 参数名
        type=int,  # 整数类型
        default=1,  # 默认1
        help="Number of generated sequences per prompt.",  # 帮助信息
    )
    parser.add_argument("--use-beam-search", action="store_true")  # 是否使用束搜索
    parser.add_argument(  # 预热迭代次数
        "--num-iters-warmup",  # 参数名
        type=int,  # 整数类型
        default=10,  # 默认10
        help="Number of iterations to run for warmup.",  # 帮助信息
    )
    parser.add_argument(  # 基准测试迭代次数
        "--num-iters", type=int, default=30, help="Number of iterations to run."  # 默认30次迭代
    )
    parser.add_argument(  # 是否启用性能分析
        "--profile",  # 参数名
        action="store_true",  # 布尔标志
        help="profile the generation process of a single batch",  # 对单批次生成过程进行性能分析
    )
    parser.add_argument(  # JSON输出路径
        "--output-json",  # 参数名
        type=str,  # 字符串类型
        default=None,  # 默认无
        help="Path to save the latency results in JSON format.",  # 保存延迟结果的JSON路径
    )
    parser.add_argument(  # 是否禁用反token化
        "--disable-detokenize",  # 参数名
        action="store_true",  # 布尔标志
        help=(  # 帮助信息
            "Do not detokenize responses (i.e. do not include "  # 不反token化响应
            "detokenization time in the latency measurement)"  # 不将反token化时间计入延迟测量
        ),
    )

    parser = EngineArgs.add_cli_args(parser)  # 添加引擎参数
    # V1 enables prefix caching by default which skews the latency
    # numbers. We need to disable prefix caching by default.
    parser.set_defaults(enable_prefix_caching=False)  # 默认禁用前缀缓存以避免影响延迟数据


def main(args: argparse.Namespace):
    """
    延迟基准测试的主函数。
    """
    engine_args = EngineArgs.from_cli_args(args)  # 从命令行参数创建引擎参数

    # Lazy import to avoid importing LLM when the bench command is not selected.
    from vllm import LLM, SamplingParams  # 延迟导入LLM和采样参数

    # NOTE(woosuk): If the request cannot be processed in a single batch,
    # the engine will automatically process the request in multiple batches.
    llm = LLM(**dataclasses.asdict(engine_args))  # 创建LLM实例
    assert llm.llm_engine.model_config.max_model_len >= (  # 断言模型最大长度足够
        args.input_len + args.output_len  # 输入长度加输出长度
    ), (
        "Please ensure that max_model_len is greater than"  # 请确保最大模型长度大于
        " the sum of input_len and output_len."  # 输入长度和输出长度之和
    )

    sampling_params = SamplingParams(  # 创建采样参数
        n=args.n,  # 每个提示生成的序列数
        temperature=1.0,  # 温度参数
        top_p=1.0,  # top-p采样参数
        ignore_eos=True,  # 忽略结束标记
        max_tokens=args.output_len,  # 最大输出token数
        detokenize=not args.disable_detokenize,  # 是否进行反token化
    )
    dummy_prompt_token_ids = np.random.randint(  # 生成随机提示token ID
        10000, size=(args.batch_size, args.input_len)  # 形状为(批次大小, 输入长度)
    )
    dummy_prompts: list[PromptType] = [  # 创建虚拟提示列表
        {"prompt_token_ids": batch} for batch in dummy_prompt_token_ids.tolist()  # 将每个批次转为列表
    ]

    def llm_generate():
        """
        执行LLM生成，支持普通生成和束搜索两种模式。
        """
        if not args.use_beam_search:  # 如果不使用束搜索
            llm.generate(dummy_prompts, sampling_params=sampling_params, use_tqdm=False)  # 普通生成
        else:  # 使用束搜索
            llm.beam_search(  # 束搜索生成
                dummy_prompts,  # 提示列表
                BeamSearchParams(  # 束搜索参数
                    beam_width=args.n,  # 束宽度
                    max_tokens=args.output_len,  # 最大token数
                    ignore_eos=True,  # 忽略结束标记
                ),
            )

    def run_to_completion(do_profile: bool = False):
        """
        运行到完成，可选择是否进行性能分析。
        """
        if do_profile:  # 如果启用性能分析
            llm.start_profile()  # 开始性能分析
            llm_generate()  # 执行生成
            llm.stop_profile()  # 停止性能分析
        else:  # 不进行性能分析
            start_time = time.perf_counter()  # 记录开始时间
            llm_generate()  # 执行生成
            end_time = time.perf_counter()  # 记录结束时间
            latency = end_time - start_time  # 计算延迟
            return latency  # 返回延迟

    print("Warming up...")  # 打印预热信息
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):  # 预热循环
        run_to_completion(do_profile=False)  # 运行预热迭代

    if args.profile:  # 如果启用性能分析
        profiler_config = engine_args.profiler_config  # 获取分析器配置
        if profiler_config.profiler == "torch":  # 如果使用torch分析器
            print(  # 打印torch分析器信息
                "Profiling with torch profiler (results will be saved to"  # 使用torch分析器
                f" {profiler_config.torch_profiler_dir})..."  # 结果保存目录
            )
        elif profiler_config.profiler == "cuda":  # 如果使用CUDA分析器
            print("Profiling with cuda profiler ...")  # 打印CUDA分析器信息
        run_to_completion(do_profile=True)  # 运行性能分析
        return  # 返回

    # Benchmark.
    latencies = []  # 初始化延迟列表
    for _ in tqdm(range(args.num_iters), desc="Bench iterations"):  # 基准测试循环
        latencies.append(run_to_completion(do_profile=False))  # 记录每次迭代的延迟
    latencies = np.array(latencies)  # 转换为NumPy数组
    percentages = [10, 25, 50, 75, 90, 99]  # 百分位数列表
    percentiles = np.percentile(latencies, percentages)  # 计算百分位数
    print(f"Avg latency: {np.mean(latencies)} seconds")  # 打印平均延迟
    for percentage, percentile in zip(percentages, percentiles):  # 遍历百分位数
        print(f"{percentage}% percentile latency: {percentile} seconds")  # 打印各百分位延迟

    # Output JSON results if specified
    if args.output_json:  # 如果指定了JSON输出路径
        results = {  # 构建结果字典
            "avg_latency": np.mean(latencies),  # 平均延迟
            "latencies": latencies.tolist(),  # 延迟列表
            "percentiles": dict(zip(percentages, percentiles.tolist())),  # 百分位数字典
        }
        with open(args.output_json, "w") as f:  # 打开文件写入
            json.dump(results, f, indent=4)  # 序列化为JSON
        save_to_pytorch_benchmark_format(args, results)  # 保存为PyTorch格式
