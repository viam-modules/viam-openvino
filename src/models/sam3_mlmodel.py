"""SAM3 MLModel resource: one active inference; concurrent calls wait in FIFO order."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, ClassVar, Dict, Mapping, Optional, Sequence, Tuple

from typing_extensions import Self
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.services.mlmodel import MLModel, Metadata, TensorInfo
from viam.utils import ValueTypes, struct_to_dict

from .config import default_cache_dir, parse_extra_config, parse_inference_precision
from .sam3_bundle import load_bundle
from .sam3_pipeline import decode_inputs


@dataclass(frozen=True)
class Sam3Attributes:
    model_dir: str
    device: str
    cache_dir: str
    inference_precision: Optional[str]
    extra_config: Dict[str, Any]


def _attributes(config: ComponentConfig) -> Sam3Attributes:
    attrs = struct_to_dict(config.attributes)
    unknown = set(attrs) - {"model_dir", "device", "cache_dir", "inference_precision", "extra_config"}
    if unknown:
        raise ValueError(f"unknown SAM3 attributes: {', '.join(sorted(unknown))}")
    directory = attrs.get("model_dir")
    if not isinstance(directory, str) or not directory:
        raise ValueError("model_dir must identify a complete SAM3 bundle")
    if not (Path(directory) / "manifest.json").is_file():
        raise ValueError("model_dir must contain manifest.json")
    device = attrs.get("device", "CPU")
    if device not in ("CPU", "GPU"):
        raise ValueError("device must be CPU or GPU; no automatic fallback is supported")
    suffix = hashlib.sha256(config.name.encode()).hexdigest()[:16]
    cache = attrs.get("cache_dir", str(Path(default_cache_dir()) / ("sam3-" + suffix)))
    if not isinstance(cache, str):
        raise ValueError("cache_dir must be a path string or empty to disable caching")
    return Sam3Attributes(directory, device, cache, parse_inference_precision(attrs), parse_extra_config(attrs))


class Sam3MLModel(MLModel, EasyResource):
    MODEL: ClassVar[str] = "viam:openvino:sam3"

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._pipeline: Optional[Any] = None
        self._metadata: Optional[Metadata] = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sam3-" + name)
        self._workers: set[Future] = set()
        self._closing = False
        self._close_task: Optional[asyncio.Task] = None
        self._device: Optional[str] = None

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        _attributes(config)
        return [], []

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        from .sam3_pipeline import Sam3Pipeline

        attrs = _attributes(config)
        bundle = load_bundle(attrs.model_dir)
        pipeline = Sam3Pipeline(bundle, device=attrs.device, cache_dir=attrs.cache_dir,
                                inference_precision=attrs.inference_precision, extra_config=attrs.extra_config)
        instance = cls(config.name)
        instance._pipeline = pipeline
        instance._device = attrs.device
        count, height, width = bundle.manifest["output_shape"]
        instance._metadata = Metadata(
            name="SAM3", type="instance_segmentation",
            description=f"SAM3 OpenVINO {attrs.device}; all proposals; native binary masks; CPU preprocessing/geometry glue",
            input_info=[TensorInfo(name="image", data_type="uint8", shape=[-1, -1, 3]),
                        TensorInfo(name="prompt", data_type="uint8", shape=[-1])],
            output_info=[TensorInfo(name="masks", data_type="uint8", shape=[count, height, width]),
                         TensorInfo(name="scores", data_type="float32", shape=[count])],
        )
        return instance

    async def infer(
        self, input_tensors: Dict[str, Any], *, extra: Optional[Mapping[str, ValueTypes]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        if self._closing:
            raise RuntimeError("SAM3 resource is closing")
        if self._pipeline is None:
            raise RuntimeError("SAM3 is not configured")
        profiling = self.logger.isEnabledFor(logging.DEBUG)
        started = perf_counter() if profiling else 0.0
        image, prompt = decode_inputs(input_tensors)
        # Like viam:openvino:mlmodel, extra calls wait FIFO on the single-worker executor.
        if profiling:
            validation_ms = (perf_counter() - started) * 1000.0
            worker = self._executor.submit(self._profiled_infer, self._pipeline, image, prompt, validation_ms)
        else:
            worker = self._executor.submit(self._pipeline.infer, image, prompt)
        self._workers.add(worker)
        future = asyncio.wrap_future(worker)

        def finished(done: asyncio.Future) -> None:
            self._workers.discard(worker)
            if not done.cancelled():
                done.exception()

        future.add_done_callback(finished)
        try:
            # Shield so a cancelled caller cannot release the pipeline while its job is still running.
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            if worker.cancelled() and self._closing:
                raise RuntimeError("SAM3 resource closed before this queued request started") from None
            worker.cancel()  # drops the job only if it has not started
            raise

    def _profiled_infer(self, pipeline, image, prompt, validation_ms):
        timings = {}
        outputs = None
        status = "error"
        error_type = None
        started = perf_counter()
        try:
            outputs = pipeline.infer(image, prompt, timings=timings)
            status = "ok"
            return outputs
        except BaseException as exc:
            error_type = type(exc).__name__
            raise
        finally:
            worker_ms = (perf_counter() - started) * 1000.0
            component_ms = sum(value for name, value in timings.items() if name != "pipeline_total")
            self.logger.debug(json.dumps({
                "event": "sam3_infer_profile", "resource": self.name, "device": self._device,
                "status": status, "error_type": error_type, "input_shape": list(image.shape),
                "input_validation_ms": validation_ms, "worker_ms": worker_ms, "stages_ms": timings,
                "unattributed_pipeline_ms": max(0.0, timings.get("pipeline_total", 0.0) - component_ms),
                "output_bytes": sum(value.nbytes for value in outputs.values()) if outputs is not None else 0,
            }))

    async def metadata(
        self, *, extra: Optional[Mapping[str, ValueTypes]] = None, timeout: Optional[float] = None,
    ) -> Metadata:
        if self._closing or self._metadata is None:
            raise RuntimeError("SAM3 metadata is unavailable: resource is not configured or is closing")
        return self._metadata

    async def close(self) -> None:
        if self._close_task is None:
            self._closing = True
            self._close_task = asyncio.create_task(self._drain_and_close())
        await asyncio.shield(self._close_task)

    async def _drain_and_close(self) -> None:
        for worker in list(self._workers):
            worker.cancel()
        # pipeline.close queues behind the running job on the single worker, so it waits for it to finish.
        try:
            if self._pipeline is not None:
                await asyncio.get_running_loop().run_in_executor(self._executor, self._pipeline.close)
        finally:
            self._pipeline = None
            await asyncio.to_thread(self._executor.shutdown, wait=True)
