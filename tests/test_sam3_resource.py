import asyncio
import threading

import numpy as np
import pytest

from models.sam3_mlmodel import Sam3MLModel


class ControlledPipeline:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.closed = threading.Event()
        self.error = None
        self.calls = 0

    def infer(self, image, prompt, *, timings=None):
        self.calls += 1
        self.started.set()
        if not self.release.wait(timeout=10):
            raise TimeoutError("test did not release the worker")
        try:
            if self.error:
                raise self.error
            return {"masks": np.ones((1, 1, 1), dtype=np.uint8), "scores": np.array([0.75], dtype=np.float32)}
        finally:
            self.finished.set()

    def close(self):
        self.closed.set()


@pytest.fixture
async def controlled_resource():
    resource = Sam3MLModel("sam3-test")
    pipeline = ControlledPipeline()
    resource._pipeline = pipeline
    yield resource, pipeline
    pipeline.release.set()
    await resource.close()


def inputs():
    return {"image": np.zeros((3, 5, 3), dtype=np.uint8), "prompt": np.frombuffer(b"bag", dtype=np.uint8)}


async def wait_started(pipeline):
    assert await asyncio.to_thread(pipeline.started.wait, 5)


async def test_concurrent_request_waits_for_running_worker_then_runs(controlled_resource):
    resource, pipeline = controlled_resource
    first = asyncio.create_task(resource.infer(inputs()))
    await wait_started(pipeline)
    second = asyncio.create_task(resource.infer(inputs()))
    await asyncio.sleep(0.05)
    assert not second.done()
    assert pipeline.calls == 1
    pipeline.release.set()
    for result in await asyncio.gather(first, second):
        np.testing.assert_array_equal(result["scores"], [0.75])
    assert pipeline.calls == 2


async def test_cancelled_running_caller_keeps_worker_and_queued_caller_is_dropped(controlled_resource):
    resource, pipeline = controlled_resource
    first = asyncio.create_task(resource.infer(inputs()))
    await wait_started(pipeline)
    queued = asyncio.create_task(resource.infer(inputs()))
    await asyncio.sleep(0)
    first.cancel()
    queued.cancel()
    for task in (first, queued):
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not pipeline.finished.is_set()
    pipeline.release.set()
    await resource.close()
    assert pipeline.finished.is_set()
    assert pipeline.calls == 1
    assert pipeline.closed.is_set()


async def test_close_rejects_queued_request_and_drains_running_one(controlled_resource):
    resource, pipeline = controlled_resource
    first = asyncio.create_task(resource.infer(inputs()))
    await wait_started(pipeline)
    queued = asyncio.create_task(resource.infer(inputs()))
    await asyncio.sleep(0)
    closing = asyncio.create_task(resource.close())
    with pytest.raises(RuntimeError, match="closed before"):
        await queued
    assert not pipeline.closed.is_set()
    pipeline.release.set()
    await asyncio.wait_for(closing, 5)
    await first
    assert pipeline.calls == 1
    assert pipeline.closed.is_set()


async def test_worker_failure_restores_admission(controlled_resource):
    resource, pipeline = controlled_resource
    pipeline.error = ValueError("model execution failed")
    pipeline.release.set()
    with pytest.raises(ValueError, match="model execution failed"):
        await resource.infer(inputs())
    pipeline.error = None
    result = await resource.infer(inputs())
    np.testing.assert_array_equal(result["scores"], [0.75])


async def test_close_drains_without_blocking_event_loop_and_rejects_new_work(controlled_resource):
    resource, pipeline = controlled_resource
    first = asyncio.create_task(resource.infer(inputs()))
    await wait_started(pipeline)
    closing = asyncio.create_task(resource.close())
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="clos"):
        await resource.infer(inputs())
    assert not pipeline.closed.is_set()
    pipeline.release.set()
    await asyncio.wait_for(closing, timeout=5)
    await first
    assert pipeline.finished.is_set()
    assert pipeline.closed.is_set()
    await resource.close()


async def test_sdk_recreation_drains_old_worker_and_reports_failed_replacement(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from test_mlmodel_resource import make_config
    from viam.module.module import Module
    from viam.proto.module import AddResourceRequest, ReconfigureResourceRequest
    from viam.services.mlmodel import MLModel

    import models.sam3_mlmodel as resource_module
    import models.sam3_pipeline as pipeline_module

    (tmp_path / "manifest.json").write_text("{}")
    old_pipeline, new_pipeline = ControlledPipeline(), ControlledPipeline()
    pipelines = iter((old_pipeline, new_pipeline))
    monkeypatch.setattr(resource_module, "load_bundle", lambda path: SimpleNamespace(manifest={"output_shape": [1, 1, 1]}))
    monkeypatch.setattr(pipeline_module, "Sam3Pipeline", lambda *args, **kwargs: next(pipelines))
    config = make_config(name="sam3-lifecycle", model_dir=str(tmp_path))
    config.model = "viam:openvino:sam3"
    config.log_configuration.level = "INFO"
    module = Module(str(tmp_path / "module.sock"))
    await module.add_resource(AddResourceRequest(config=config))
    name = MLModel.get_resource_name(config.name)
    old = module.server.get_resource(Sam3MLModel, name)
    first = asyncio.create_task(old.infer(inputs()))
    await wait_started(old_pipeline)
    replacement = asyncio.create_task(module.reconfigure_resource(ReconfigureResourceRequest(config=config)))
    try:
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="clos"):
            await old.infer(inputs())
        assert not old_pipeline.closed.is_set()
        old_pipeline.release.set()
        await asyncio.wait_for(replacement, 5)
        await first
        assert old_pipeline.finished.is_set()
        assert old_pipeline.closed.is_set()
        current = module.server.get_resource(Sam3MLModel, name)
        assert current is not old
        new_pipeline.release.set()
        result = await current.infer(inputs())
        np.testing.assert_array_equal(result["scores"], [0.75])
        config.attributes.fields["model_dir"].string_value = str(tmp_path / "missing")
        with pytest.raises(ValueError, match="model_dir"):
            await module.reconfigure_resource(ReconfigureResourceRequest(config=config))
        assert new_pipeline.closed.is_set()
        with pytest.raises(RuntimeError, match="clos"):
            await current.infer(inputs())
    finally:
        old_pipeline.release.set()
        new_pipeline.release.set()
        await old.close()
        if "current" in locals():
            await current.close()


async def test_cancelled_close_caller_does_not_abandon_worker_cleanup(controlled_resource):
    resource, pipeline = controlled_resource
    first = asyncio.create_task(resource.infer(inputs()))
    await wait_started(pipeline)
    closing = asyncio.create_task(resource.close())
    await asyncio.sleep(0)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    with pytest.raises(RuntimeError, match="clos"):
        await resource.infer(inputs())
    assert not pipeline.closed.is_set()
    pipeline.release.set()
    await asyncio.wait_for(resource.close(), 5)
    await first
    assert pipeline.finished.is_set()
    assert pipeline.closed.is_set()


def sam3_config(tmp_path, **attrs):
    from test_mlmodel_resource import make_config

    (tmp_path / "manifest.json").write_text("{}")
    return make_config(name="sam3-attrs", model_dir=str(tmp_path), device="GPU", **attrs)


def test_precision_and_extra_config_attributes_are_parsed(tmp_path):
    from models.sam3_mlmodel import _attributes

    attrs = _attributes(sam3_config(tmp_path, inference_precision="F32",
                                    extra_config={"GPU_QUEUE_PRIORITY": "HIGH", "NUM_STREAMS": 1}))
    assert attrs.inference_precision == "f32"
    assert attrs.extra_config == {"GPU_QUEUE_PRIORITY": "HIGH", "NUM_STREAMS": 1}
    assert isinstance(attrs.extra_config["NUM_STREAMS"], int)


@pytest.mark.parametrize("attrs,match", [
    ({"inference_precision": "int8"}, "inference_precision"),
    ({"extra_config": "GPU_QUEUE_PRIORITY=HIGH"}, "extra_config"),
    ({"num_requests": 2}, "unknown SAM3 attributes"),
])
def test_invalid_speed_attributes_fail_at_validation(tmp_path, attrs, match):
    with pytest.raises(ValueError, match=match):
        Sam3MLModel.validate_config(sam3_config(tmp_path, **attrs))
