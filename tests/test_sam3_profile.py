import json

import numpy as np
import openvino as ov
import pytest

from conftest import build_tiny_ov_model
from models import sam3_pipeline
from models.sam3_pipeline import Sam3Pipeline, _profile_stage
from test_sam3_resource import controlled_resource, inputs, wait_started  # noqa: F401


def test_real_network_profile_records_synchronous_execution_without_changing_output(monkeypatch):
    compiled = ov.Core().compile_model(build_tiny_ov_model(np.ones((4, 3, 3, 3), dtype=np.float32)), "CPU")
    pipeline = Sam3Pipeline.__new__(Sam3Pipeline)
    pipeline._compiled = {"image_encoder": compiled}
    pipeline._requests = {"image_encoder": compiled.create_infer_request()}
    clock = iter((10.0, 10.125))
    monkeypatch.setattr(sam3_pipeline, "perf_counter", lambda: next(clock))
    timings = {}
    output = pipeline._run("image_encoder", [np.ones((1, 3, 8, 8), dtype=np.float32)], timings=timings)[0]
    np.testing.assert_array_equal(output[0, :, 3, 3], [27, 27, 27, 27])
    assert timings == {"image_encoder": 125.0}


def test_failed_stage_keeps_its_elapsed_time_and_original_error(monkeypatch):
    clock = iter((3.0, 3.25))
    monkeypatch.setattr(sam3_pipeline, "perf_counter", lambda: next(clock))
    timings = {}
    with pytest.raises(ValueError, match="stage failed"):
        with _profile_stage(timings, "decoder"):
            raise ValueError("stage failed")
    assert timings == {"decoder": 250.0}


def profiles(caplog):
    reports = []
    for record in caplog.records:
        try:
            data = json.loads(record.getMessage())
        except (ValueError, TypeError):
            continue
        if data.get("event") == "sam3_infer_profile":
            reports.append(data)
    return reports


async def test_debug_level_profiles_without_changing_call_or_public_output(controlled_resource, caplog):
    resource, pipeline = controlled_resource
    pipeline.release.set()
    caplog.set_level("DEBUG", logger=resource.logger.name)
    result = await resource.infer(inputs())
    np.testing.assert_array_equal(result["scores"], [0.75])
    assert set(result) == {"masks", "scores"}
    report, = profiles(caplog)
    assert report["status"] == "ok"
    assert report["input_shape"] == [3, 5, 3]
    assert report["output_bytes"] == 5
    assert report["worker_ms"] >= 0
    assert report["input_validation_ms"] >= 0
    assert "prompt" not in report and "image" not in report


async def test_info_level_does_not_emit_profile(controlled_resource, caplog):
    resource, pipeline = controlled_resource
    pipeline.release.set()
    caplog.set_level("INFO", logger=resource.logger.name)
    await resource.infer(inputs())
    assert profiles(caplog) == []


async def test_worker_failure_emits_failed_profile_without_masking_error(controlled_resource, caplog):
    resource, pipeline = controlled_resource
    pipeline.error = ValueError("model execution failed")
    pipeline.release.set()
    caplog.set_level("DEBUG", logger=resource.logger.name)
    with pytest.raises(ValueError, match="model execution failed"):
        await resource.infer(inputs())
    report, = profiles(caplog)
    assert report["status"] == "error"
    assert report["error_type"] == "ValueError"
    assert report["output_bytes"] == 0


async def test_cancelled_caller_still_gets_worker_completion_profile(controlled_resource, caplog):
    import asyncio

    resource, pipeline = controlled_resource
    caplog.set_level("DEBUG", logger=resource.logger.name)
    call = asyncio.create_task(resource.infer(inputs()))
    await wait_started(pipeline)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    assert profiles(caplog) == []
    pipeline.release.set()
    await resource.close()
    report, = profiles(caplog)
    assert report["status"] == "ok"
    assert report["output_bytes"] == 5
