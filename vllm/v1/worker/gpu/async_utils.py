# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib  # 导入上下文管理器工具模块

import numpy as np  # 导入 NumPy 数值计算库
import torch  # 导入 PyTorch 深度学习框架

from vllm.v1.outputs import AsyncModelRunnerOutput, LogprobsTensors, ModelRunnerOutput  # 从 vllm 输出模块导入异步模型运行输出、对数概率张量和模型运行输出类
from vllm.v1.worker.gpu.sample.output import SamplerOutput  # 从采样输出模块导入采样器输出类


# 异步输出类：利用独立的 CUDA 拷贝流将采样结果（token ID、logprobs 等）
# 从 GPU 异步传输到 CPU，避免阻塞主计算流，从而实现计算与数据传输的流水线重叠。
class AsyncOutput(AsyncModelRunnerOutput):
    # 初始化异步输出对象，设置拷贝流并启动异步传输
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,  # 模型运行器的输出结果
        sampler_output: SamplerOutput,  # 采样器的输出结果
        num_sampled_tokens: torch.Tensor,  # 每个请求采样的 token 数量张量
        main_stream: torch.cuda.Stream,  # 主计算 CUDA 流
        copy_stream: torch.cuda.Stream,  # 用于异步拷贝的 CUDA 流
        copy_event: torch.cuda.Event,  # 用于同步拷贝完成的 CUDA 事件
    ):
        # NOTE(woosuk): We must retain references to the GPU tensors,
        # as the copy operations are performed on a different CUDA stream than
        # the one where the tensors were created.
        self.model_runner_output = model_runner_output  # 保存模型运行输出的引用，防止 GPU 张量被回收
        self.sampler_output = sampler_output  # 保存采样器输出的引用
        self.num_sampled_tokens = num_sampled_tokens  # 保存采样 token 数量张量的引用
        self.copy_event = copy_event  # 保存拷贝事件用于后续同步

        with stream(copy_stream, main_stream):  # 切换到拷贝流执行以下操作
            copy_stream.wait_stream(main_stream)  # 等待主流上的计算完成后再开始拷贝

            self.sampled_token_ids = async_copy_to_np(sampler_output.sampled_token_ids)  # 异步将采样的 token ID 从 GPU 拷贝到 CPU NumPy 数组
            self.logprobs_tensors: LogprobsTensors | None = None  # 初始化对数概率张量为 None
            if sampler_output.logprobs_tensors is not None:  # 如果存在对数概率张量
                self.logprobs_tensors = (  # 异步将对数概率张量拷贝到 CPU
                    sampler_output.logprobs_tensors.to_cpu_nonblocking()
                )
            self.num_nans: np.ndarray | None = None  # 初始化 NaN 计数数组为 None
            if sampler_output.num_nans is not None:  # 如果存在 NaN 计数
                self.num_nans = async_copy_to_np(sampler_output.num_nans)  # 异步将 NaN 计数拷贝到 CPU
            self.num_sampled_tokens_np = async_copy_to_np(num_sampled_tokens)  # 异步将采样 token 数量拷贝到 CPU
            self.prompt_logprobs_dict = {  # 异步将提示词对数概率字典中的张量拷贝到 CPU
                k: v.to_cpu_nonblocking() if v is not None else None
                for k, v in self.model_runner_output.prompt_logprobs_dict.items()
            }
            self.copy_event.record(copy_stream)  # 在拷贝流上记录事件，标记所有拷贝操作完成

    # 同步拷贝事件并将 NumPy 数组转换为 Python 列表，组装最终的模型输出。
    def get_output(self) -> ModelRunnerOutput:
        self.copy_event.synchronize()  # 等待所有异步拷贝操作完成

        # NOTE(woosuk): The following code is to ensure compatibility with
        # the existing model runner.
        # Going forward, we should keep the data structures as NumPy arrays
        # rather than Python lists.
        sampled_token_ids: list[list[int]] = self.sampled_token_ids.tolist()  # 将采样 token ID 的 NumPy 数组转换为 Python 嵌套列表
        num_sampled_tokens: list[int] = self.num_sampled_tokens_np.tolist()  # 将采样 token 数量转换为 Python 列表
        for token_ids, num_tokens in zip(sampled_token_ids, num_sampled_tokens):  # 遍历每个请求的 token ID 和实际采样数
            del token_ids[num_tokens:]  # 截断多余的填充 token，只保留实际采样的 token
        self.model_runner_output.sampled_token_ids = sampled_token_ids  # 将处理后的 token ID 赋值给输出

        if self.num_nans is not None:  # 如果存在 NaN 计数信息
            self.model_runner_output.num_nans_in_logits = dict(  # 构建请求 ID 到 NaN 数量的映射字典
                zip(self.model_runner_output.req_ids, self.num_nans.tolist())
            )

        if self.logprobs_tensors is not None:  # 如果存在对数概率张量
            self.model_runner_output.logprobs = self.logprobs_tensors.tolists()  # 将对数概率张量转换为嵌套列表
        self.model_runner_output.prompt_logprobs_dict = self.prompt_logprobs_dict  # 赋值提示词对数概率字典
        return self.model_runner_output  # 返回完整的模型运行输出


# 异步池化输出类：与 AsyncOutput 类似，但专用于池化模型（如嵌入模型）。
# 将池化层的输出张量异步拷贝到 CPU，并处理无效结果的掩码。
class AsyncPoolingOutput(AsyncModelRunnerOutput):
    # 初始化异步池化输出对象，设置拷贝流并启动异步传输
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,  # 模型运行器的输出结果
        pooler_output: torch.Tensor,  # 池化层输出张量
        is_valid: torch.Tensor | None,  # 有效性掩码张量，标记哪些请求的结果有效
        main_stream: torch.cuda.Stream,  # 主计算 CUDA 流
        copy_stream: torch.cuda.Stream,  # 用于异步拷贝的 CUDA 流
        copy_event: torch.cuda.Event,  # 用于同步拷贝完成的 CUDA 事件
    ):
        self.model_runner_output = model_runner_output  # 保存模型运行输出的引用
        self.pooler_output = pooler_output  # 保存池化输出张量的引用
        self.is_valid = is_valid  # 保存有效性掩码的引用
        self.copy_event = copy_event  # 保存拷贝事件用于后续同步

        with stream(copy_stream, main_stream):  # 切换到拷贝流执行以下操作
            copy_stream.wait_stream(main_stream)  # 等待主流上的计算完成
            self.pooler_output_cpu = self.pooler_output.to("cpu", non_blocking=True)  # 异步将池化输出拷贝到 CPU
            if self.is_valid is not None:  # 如果存在有效性掩码
                self.is_valid_cpu = self.is_valid.to("cpu", non_blocking=True)  # 异步将有效性掩码拷贝到 CPU
            else:  # 如果没有有效性掩码
                self.is_valid_cpu = None  # 设置为 None
            self.copy_event.record(copy_stream)  # 在拷贝流上记录事件，标记拷贝完成

    # 同步拷贝事件，将池化输出拆分为每个请求的独立张量，并将无效请求标记为 None。
    def get_output(self) -> ModelRunnerOutput:
        pooler_output = list(self.pooler_output_cpu.unbind(dim=0))  # 将批次维度拆分为独立张量的列表
        self.copy_event.synchronize()  # 等待异步拷贝完成
        if self.is_valid_cpu is not None:  # 如果存在有效性掩码
            is_valid_cpu = self.is_valid_cpu.tolist()  # 将有效性掩码转换为 Python 列表
            for i, is_valid in enumerate(is_valid_cpu):  # 遍历每个请求的有效性标记
                if not is_valid:  # 如果该请求的结果无效
                    pooler_output[i] = None  # 将无效请求的输出设为 None
        self.model_runner_output.pooler_output = pooler_output  # 将处理后的池化输出赋值给模型输出
        return self.model_runner_output  # 返回完整的模型运行输出


# 将 GPU 张量异步拷贝到 CPU 并转换为 NumPy 数组（非阻塞操作）。
def async_copy_to_np(x: torch.Tensor) -> np.ndarray:
    return x.to("cpu", non_blocking=True).numpy()  # 非阻塞地将张量拷贝到 CPU 并转换为 NumPy 数组


# 轻量级 CUDA 流切换上下文管理器：切换到目标流执行操作，退出时恢复原始流。
# 相比 torch.cuda.stream() 避免了当前流和设备的额外查询开销。
@contextlib.contextmanager  # 标记为上下文管理器
def stream(to_stream: torch.cuda.Stream, from_stream: torch.cuda.Stream):
    """Lightweight version of torch.cuda.stream() context manager which
    avoids current_stream and device lookups.
    """
    try:  # 尝试切换流
        torch.cuda.set_stream(to_stream)  # 将当前 CUDA 流切换到目标流
        yield  # 在目标流中执行用户代码
    finally:  # 无论是否发生异常都恢复原始流
        torch.cuda.set_stream(from_stream)  # 恢复到原始 CUDA 流
