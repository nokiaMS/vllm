# SPDX-License-Identifier: Apache-2.0  # Apache-2.0 开源协议标识
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project  # 版权声明：vLLM项目贡献者
import argparse  # 导入命令行参数解析模块

from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG  # 导入vLLM子命令解析器附录

from .plot import SweepPlotArgs  # 导入扫描绘图参数类
from .plot import main as plot_main  # 导入绘图主函数
from .plot_pareto import SweepPlotParetoArgs  # 导入帕累托绘图参数类
from .plot_pareto import main as plot_pareto_main  # 导入帕累托绘图主函数
from .serve import SweepServeArgs  # 导入扫描服务参数类
from .serve import main as serve_main  # 导入服务主函数
from .serve_workload import SweepServeWorkloadArgs  # 导入工作负载扫描参数类
from .serve_workload import main as serve_workload_main  # 导入工作负载扫描主函数
from .startup import SweepStartupArgs  # 导入启动扫描参数类
from .startup import main as startup_main  # 导入启动扫描主函数

SUBCOMMANDS = (  # 子命令元组，包含参数类和入口函数的配对
    (SweepServeArgs, serve_main),  # 服务基准测试扫描
    (SweepServeWorkloadArgs, serve_workload_main),  # 工作负载服务基准测试扫描
    (SweepStartupArgs, startup_main),  # 启动时间基准测试扫描
    (SweepPlotArgs, plot_main),  # 绘制性能曲线图
    (SweepPlotParetoArgs, plot_pareto_main),  # 绘制帕累托前沿图
)


def add_cli_args(parser: argparse.ArgumentParser):
    """
    添加扫描基准测试的命令行子命令参数。
    """
    subparsers = parser.add_subparsers(required=True, dest="sweep_type")  # 创建必需的子命令解析器

    for cmd, entrypoint in SUBCOMMANDS:  # 遍历所有子命令
        cmd_subparser = subparsers.add_parser(  # 为每个子命令创建解析器
            cmd.parser_name,  # 子命令名称
            description=cmd.parser_help,  # 子命令描述
            usage=f"vllm bench sweep {cmd.parser_name} [options]",  # 用法说明
        )
        cmd_subparser.set_defaults(dispatch_function=entrypoint)  # 设置默认的分发函数
        cmd.add_cli_args(cmd_subparser)  # 添加子命令特有的参数
        cmd_subparser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(  # 设置附录
            subcmd=f"sweep {cmd.parser_name}"  # 格式化子命令名
        )


def main(args: argparse.Namespace):
    """
    扫描基准测试的主函数，根据子命令分发到对应的处理函数。
    """
    args.dispatch_function(args)  # 调用子命令对应的处理函数
