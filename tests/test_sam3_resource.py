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

    def infer(self, image, prompt):
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


async def test_second_request_is_rejected_while_first_worker_runs(controlled_resource):
    resource, pipeline = controlled_resource
    first = asyncio.create_task(resource.infer(inputs()))
    await wait_started(pipeline)
    with pytest.raises(RuntimeError, match="busy"):
        await resource.infer(inputs())
    pipeline.release.set()
    result = await first
    np.testing.assert_array_equal(result["scores"], [0.75])


async def test_cancelled_caller_does_not_free_active_worker_capacity(controlled_resource):
    resource, pipeline = controlled_resource
    first = asyncio.create_task(resource.infer(inputs()))
    await wait_started(pipeline)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    with pytest.raises(RuntimeError, match="busy"):
        await resource.infer(inputs())
    pipeline.release.set()
    await resource.close()
    assert pipeline.finished.is_set()
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
