# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import gc
import json
import os
import pathlib
import subprocess
import sys
from typing import Any

import pytest
import torch

import vllm.model_executor.model_loader.tensorizer
from tests.utils import VLLM_PATH, RemoteOpenAIServer
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.model_loader.tensorizer import (
    TensorizerConfig,
    TensorSerializer,
    is_vllm_tensorized,
    open_stream,
    tensorize_vllm_model,
)
from vllm.model_executor.model_loader.tensorizer_loader import (
    BLACKLISTED_TENSORIZER_ARGS,
)
from vllm.utils.import_utils import PlaceholderModule

from .conftest import DummyExecutor, assert_from_collective_rpc

try:
    import tensorizer
    from tensorizer import EncryptionParams
except ImportError:
    tensorizer = PlaceholderModule("tensorizer")  # type: ignore[assignment]
    EncryptionParams = tensorizer.placeholder_attr("EncryptionParams")


# 测试 Tensorizer 模型序列化/反序列化的正确性，包括加密、分片、参数传递等场景

class TensorizerCaughtError(Exception):
    pass


EXAMPLES_PATH = VLLM_PATH / "examples"

pytest_plugins = ("pytest_asyncio",)

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, seed=0)


def patch_init_and_catch_error(self, obj, method_name, expected_error: type[Exception]):
    original = getattr(obj, method_name, None)
    if original is None:
        raise ValueError("Method '{}' not found.".format(method_name))

    def wrapper(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except expected_error as err:
            raise TensorizerCaughtError from err

    setattr(obj, method_name, wrapper)

    self.load_model()


def assert_specific_tensorizer_error_is_raised(
    executor,
    obj: Any,
    method_name: str,
    expected_error: type[Exception],
):
    with pytest.raises(TensorizerCaughtError):
        executor.collective_rpc(
            patch_init_and_catch_error,
            args=(
                obj,
                method_name,
                expected_error,
            ),
        )


def is_curl_installed():
    try:
        subprocess.check_call(["curl", "--version"])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def write_keyfile(keyfile_path: str):
    encryption_params = EncryptionParams.random()
    pathlib.Path(keyfile_path).parent.mkdir(parents=True, exist_ok=True)
    with open(keyfile_path, "wb") as f:
        f.write(encryption_params.key)


@pytest.mark.skipif(not is_curl_installed(), reason="cURL is not installed")
# 测试加密后序列化再反序列化的模型输出是否一致
def test_deserialized_encrypted_vllm_model_has_same_outputs(
    model_ref, vllm_runner, tmp_path, model_path
):
    args = EngineArgs(model=model_ref)
    with vllm_runner(model_ref) as vllm_model:
        key_path = tmp_path / model_ref / "model.key"
        write_keyfile(key_path)

        outputs = vllm_model.generate(prompts, sampling_params)

    config_for_serializing = TensorizerConfig(
        tensorizer_uri=str(model_path), encryption_keyfile=str(key_path)
    )

    tensorize_vllm_model(args, config_for_serializing)

    config_for_deserializing = TensorizerConfig(
        tensorizer_uri=str(model_path), encryption_keyfile=str(key_path)
    )

    with vllm_runner(
        model_ref,
        load_format="tensorizer",
        model_loader_extra_config=config_for_deserializing,
    ) as loaded_vllm_model:  # noqa: E501
        deserialized_outputs = loaded_vllm_model.generate(prompts, sampling_params)
        # noqa: E501

        assert outputs == deserialized_outputs


# 测试从 HuggingFace 模型序列化后反序列化的输出是否一致
def test_deserialized_hf_model_has_same_outputs(
    hf_runner, vllm_runner, tmp_path, model_ref, model_path
):
    with hf_runner(model_ref) as hf_model:
        max_tokens = 50
        outputs = hf_model.generate_greedy(prompts, max_tokens=max_tokens)
        with open_stream(model_path, "wb+") as stream:
            serializer = TensorSerializer(stream)
            serializer.write_module(hf_model.model)

    with vllm_runner(
        model_ref,
        load_format="tensorizer",
        model_loader_extra_config=TensorizerConfig(
            tensorizer_uri=str(model_path),
            num_readers=1,
        ),
    ) as loaded_hf_model:
        deserialized_outputs = loaded_hf_model.generate_greedy(
            prompts, max_tokens=max_tokens
        )

        assert outputs == deserialized_outputs


# 测试未指定 tensorizer 加载格式时传入额外配置应抛出错误
def test_load_without_tensorizer_load_format(vllm_runner, capfd, model_ref):
    model = None
    try:
        model = vllm_runner(
            model_ref, model_loader_extra_config=TensorizerConfig(tensorizer_uri="test")
        )
        pytest.fail("Expected RuntimeError for extra config keys")
    except RuntimeError:
        out, err = capfd.readouterr()
        combined_output = out + err
        assert (
            "ValueError: Unexpected extra config keys for load format auto"
        ) in combined_output
    finally:
        del model
        gc.collect()
        torch.accelerator.empty_cache()


# 测试使用无效加载格式时应抛出 ValueError
def test_raise_value_error_on_invalid_load_format(vllm_runner, capfd, model_ref):
    model = None
    try:
        model = vllm_runner(
            model_ref,
            load_format="safetensors",
            model_loader_extra_config=TensorizerConfig(tensorizer_uri="test"),
        )
        pytest.fail("Expected RuntimeError for extra config keys")
    except RuntimeError:
        out, err = capfd.readouterr()

        combined_output = out + err
        assert (
            "ValueError: Unexpected extra config keys for load format safetensors"
        ) in combined_output
    finally:
        del model
        gc.collect()
        torch.accelerator.empty_cache()


@pytest.mark.skipif(torch.accelerator.device_count() < 2, reason="Requires 2 GPUs")
# 测试张量并行模式下未使用路径模板时应报错
def test_tensorizer_with_tp_path_without_template(vllm_runner, capfd):
    try:
        model_ref = "EleutherAI/pythia-1.4b"
        tensorized_path = f"s3://tensorized/{model_ref}/fp16/model.tensors"

        vllm_runner(
            model_ref,
            load_format="tensorizer",
            model_loader_extra_config=TensorizerConfig(
                tensorizer_uri=tensorized_path,
                num_readers=1,
                s3_endpoint="object.ord1.coreweave.com",
            ),
            tensor_parallel_size=2,
            disable_custom_all_reduce=True,
        )
    except RuntimeError:
        out, err = capfd.readouterr()
        combined_output = out + err
        assert (
            "ValueError: For a sharded model, tensorizer_uri "
            "should include a string format template like '%04d' "
            "to be formatted with the rank "
            "of the shard"
        ) in combined_output


@pytest.mark.skipif(torch.accelerator.device_count() < 2, reason="Requires 2 GPUs")
# 测试张量并行下加密序列化/反序列化的模型输出一致性
def test_deserialized_encrypted_vllm_model_with_tp_has_same_outputs(
    vllm_runner, tmp_path
):
    model_ref = "EleutherAI/pythia-1.4b"
    # record outputs from un-sharded un-tensorized model
    with vllm_runner(
        model_ref,
        disable_custom_all_reduce=True,
        enforce_eager=True,
    ) as base_model:
        outputs = base_model.generate(prompts, sampling_params)

    # load model with two shards and serialize with encryption
    model_path = str(tmp_path / model_ref / "model-%02d.tensors")
    key_path = tmp_path / (model_ref + ".key")

    tensorizer_config = TensorizerConfig(
        tensorizer_uri=model_path,
        encryption_keyfile=str(key_path),
    )

    tensorize_vllm_model(
        engine_args=EngineArgs(
            model=model_ref,
            tensor_parallel_size=2,
            disable_custom_all_reduce=True,
            enforce_eager=True,
        ),
        tensorizer_config=tensorizer_config,
    )
    assert os.path.isfile(model_path % 0), "Serialization subprocess failed"
    assert os.path.isfile(model_path % 1), "Serialization subprocess failed"

    with vllm_runner(
        model_ref,
        tensor_parallel_size=2,
        load_format="tensorizer",
        disable_custom_all_reduce=True,
        enforce_eager=True,
        model_loader_extra_config=tensorizer_config,
    ) as loaded_vllm_model:
        deserialized_outputs = loaded_vllm_model.generate(prompts, sampling_params)

    assert outputs == deserialized_outputs


@pytest.mark.flaky(reruns=3)
# 测试 vLLM 序列化后的模型输出与原始模型一致
def test_vllm_tensorized_model_has_same_outputs(
    model_ref, vllm_runner, tmp_path, model_path
):
    gc.collect()
    torch.accelerator.empty_cache()
    config = TensorizerConfig(tensorizer_uri=str(model_path))
    args = EngineArgs(model=model_ref)

    with vllm_runner(model_ref) as vllm_model:
        outputs = vllm_model.generate(prompts, sampling_params)

    tensorize_vllm_model(args, config)
    assert is_vllm_tensorized(config)

    with vllm_runner(
        model_ref, load_format="tensorizer", model_loader_extra_config=config
    ) as loaded_vllm_model:
        deserialized_outputs = loaded_vllm_model.generate(prompts, sampling_params)
        # noqa: E501

        assert outputs == deserialized_outputs


# 测试仅使用模型张量文件进行加载的向后兼容性
def test_load_with_just_model_tensors(just_serialize_model_tensors, model_ref):
    # For backwards compatibility, ensure Tensorizer can be still be loaded
    # for inference by passing the model reference name, not a local/S3 dir,
    # and the location of the model tensors

    model_dir = just_serialize_model_tensors

    extra_config = {"tensorizer_uri": f"{model_dir}/model.tensors"}

    ## Start OpenAI API server
    args = [
        "--load-format",
        "tensorizer",
        "--model-loader-extra-config",
        json.dumps(extra_config),
    ]

    with RemoteOpenAIServer(model_ref, args):
        # This test only concerns itself with being able to load the model
        # and successfully initialize the server
        pass


# 测试序列化参数是否正确传递给 TensorSerializer
def test_assert_serialization_kwargs_passed_to_tensor_serializer(tmp_path):
    serialization_params = {
        "limit_cpu_concurrency": 2,
    }
    model_ref = "facebook/opt-125m"
    model_path = tmp_path / (model_ref + ".tensors")
    config = TensorizerConfig(
        tensorizer_uri=str(model_path), serialization_kwargs=serialization_params
    )
    llm = LLM(
        model=model_ref,
    )

    def serialization_test(self, *args, **kwargs):
        # This is performed in the ephemeral worker process, so monkey-patching
        # will actually work, and cleanup is guaranteed so don't
        # need to reset things

        original_dict = serialization_params
        to_compare = {}

        original = tensorizer.serialization.TensorSerializer.__init__

        def tensorizer_serializer_wrapper(self, *args, **kwargs):
            nonlocal to_compare
            to_compare = kwargs.copy()
            return original(self, *args, **kwargs)

        tensorizer.serialization.TensorSerializer.__init__ = (
            tensorizer_serializer_wrapper
        )

        tensorizer_config = TensorizerConfig(**kwargs["tensorizer_config"])
        self.save_tensorized_model(
            tensorizer_config=tensorizer_config,
        )
        return to_compare | original_dict == to_compare

    kwargs = {"tensorizer_config": config.to_serializable()}

    assert assert_from_collective_rpc(llm, serialization_test, kwargs)


# 测试反序列化参数是否正确传递给 TensorDeserializer
def test_assert_deserialization_kwargs_passed_to_tensor_deserializer(tmp_path, capfd):
    deserialization_kwargs = {
        "num_readers": "bar",  # illegal value
    }

    serialization_params = {
        "limit_cpu_concurrency": 2,
    }

    model_ref = "facebook/opt-125m"
    model_path = tmp_path / (model_ref + ".tensors")
    config = TensorizerConfig(
        tensorizer_uri=str(model_path), serialization_kwargs=serialization_params
    )

    args = EngineArgs(model=model_ref)
    tensorize_vllm_model(args, config)

    loader_tc = TensorizerConfig(
        tensorizer_uri=str(model_path),
        deserialization_kwargs=deserialization_kwargs,
    )

    engine_args = EngineArgs(
        model="facebook/opt-125m",
        load_format="tensorizer",
        model_loader_extra_config=loader_tc.to_serializable(),
    )

    vllm_config = engine_args.create_engine_config()
    executor = DummyExecutor(vllm_config)

    assert_specific_tensorizer_error_is_raised(
        executor,
        tensorizer.serialization.TensorDeserializer,
        "__init__",
        TypeError,
    )


# 测试流参数是否正确传递给 open_stream
def test_assert_stream_kwargs_passed_to_tensor_deserializer(tmp_path, capfd):
    deserialization_kwargs = {
        "num_readers": 1,
    }

    serialization_params = {
        "limit_cpu_concurrency": 2,
    }

    model_ref = "facebook/opt-125m"
    model_path = tmp_path / (model_ref + ".tensors")
    config = TensorizerConfig(
        tensorizer_uri=str(model_path), serialization_kwargs=serialization_params
    )

    args = EngineArgs(model=model_ref)
    tensorize_vllm_model(args, config)

    stream_kwargs = {"mode": "foo"}

    loader_tc = TensorizerConfig(
        tensorizer_uri=str(model_path),
        deserialization_kwargs=deserialization_kwargs,
        stream_kwargs=stream_kwargs,
    )

    engine_args = EngineArgs(
        model="facebook/opt-125m",
        load_format="tensorizer",
        model_loader_extra_config=loader_tc.to_serializable(),
    )

    vllm_config = engine_args.create_engine_config()
    executor = DummyExecutor(vllm_config)

    assert_specific_tensorizer_error_is_raised(
        executor,
        vllm.model_executor.model_loader.tensorizer,
        "open_stream",
        ValueError,
    )


@pytest.mark.asyncio
# 测试序列化和服务入口点的端到端流程
async def test_serialize_and_serve_entrypoints(tmp_path):
    model_ref = "facebook/opt-125m"

    suffix = "test"
    try:
        result = subprocess.run(
            [
                sys.executable,
                f"{VLLM_PATH}/examples/others/tensorize_vllm_model.py",
                "--model",
                model_ref,
                "serialize",
                "--serialized-directory",
                str(tmp_path),
                "--suffix",
                suffix,
                "--serialization-kwargs",
                '{"limit_cpu_concurrency": 4}',
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print("Tensorizing failed.")
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)
        raise

    assert "Successfully serialized" in result.stdout

    # Next, try to serve with vllm serve
    model_uri = tmp_path / "vllm" / model_ref / suffix / "model.tensors"

    model_loader_extra_config = {
        "tensorizer_uri": str(model_uri),
        "stream_kwargs": {
            "force_http": False,
        },
        "deserialization_kwargs": {
            "verify_hash": True,
            "num_readers": 8,
        },
    }

    cmd = [
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        "--host",
        "localhost",
        "--load-format",
        "tensorizer",
        model_ref,
        "--model-loader-extra-config",
        json.dumps(model_loader_extra_config, indent=2),
    ]

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    assert proc.stdout is not None
    fut = proc.stdout.readuntil(b"Application startup complete.")

    try:
        await asyncio.wait_for(fut, 180)
    except asyncio.TimeoutError:
        pytest.fail("Server did not start successfully")
    finally:
        proc.terminate()
    await proc.communicate()


@pytest.mark.parametrize("illegal_value", BLACKLISTED_TENSORIZER_ARGS)
# 测试黑名单参数在加载时应被拒绝
def test_blacklisted_parameter_for_loading(tmp_path, vllm_runner, capfd, illegal_value):
    serialization_params = {
        "limit_cpu_concurrency": 2,
    }

    model_ref = "facebook/opt-125m"
    model_path = tmp_path / (model_ref + ".tensors")
    config = TensorizerConfig(
        tensorizer_uri=str(model_path), serialization_kwargs=serialization_params
    )

    args = EngineArgs(model=model_ref)
    tensorize_vllm_model(args, config)

    loader_tc = {"tensorizer_uri": str(model_path), illegal_value: "foo"}

    try:
        vllm_runner(
            model_ref,
            load_format="tensorizer",
            model_loader_extra_config=loader_tc,
        )
    except RuntimeError:
        out, err = capfd.readouterr()
        combined_output = out + err
        assert (
            f"ValueError: {illegal_value} is not an allowed Tensorizer argument."
        ) in combined_output
