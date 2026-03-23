# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.llm_engine import LLMEngine


# [中文注释] 用于测试调度器插件注入的虚拟V1调度器，schedule()时抛出异常
class DummyV1Scheduler(Scheduler):
    def schedule(self):
        raise Exception("Exception raised by DummyV1Scheduler")


# [中文注释] 测试V1自定义调度器插件是否被正确加载和调用
def test_scheduler_plugins_v1(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as m:
        # Explicitly turn off engine multiprocessing so
        # that the scheduler runs in this process
        m.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

        with pytest.raises(Exception) as exception_info:
            engine_args = EngineArgs(
                model="facebook/opt-125m",
                enforce_eager=True,  # reduce test time
                scheduler_cls=DummyV1Scheduler,
            )

            engine = LLMEngine.from_engine_args(engine_args=engine_args)

            sampling_params = SamplingParams(max_tokens=1)
            engine.add_request("0", "foo", sampling_params)
            engine.step()

        assert str(exception_info.value) == "Exception raised by DummyV1Scheduler"
