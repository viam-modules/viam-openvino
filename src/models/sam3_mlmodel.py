"""SAM3 MLModel resource with one active inference and no waiting queue."""

from __future__ import annotations

import asyncio
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar, Dict, Mapping, Optional, Sequence, Tuple

from typing_extensions import Self
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.services.mlmodel import MLModel, Metadata, TensorInfo
from viam.utils import ValueTypes, struct_to_dict

from .config import default_cache_dir
from .sam3_bundle import load_bundle
from .sam3_pipeline import decode_inputs


def _attributes(config: ComponentConfig) -> tuple[str, str, str]:
    attrs = struct_to_dict(config.attributes)
    unknown = set(attrs) - {"model_dir", "device", "cache_dir"}
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
    return directory, device, cache


class Sam3MLModel(MLModel, EasyResource):
    MODEL: ClassVar[str] = "viam:openvino:sam3"

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._pipeline: Optional[Any] = None
        self._metadata: Optional[Metadata] = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sam3-" + name)
        self._active: Optional[asyncio.Future] = None
        self._closing = False
        self._close_task: Optional[asyncio.Task] = None

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        _attributes(config)
        return [], []

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        from .sam3_pipeline import Sam3Pipeline

        directory, device, cache = _attributes(config)
        bundle = load_bundle(directory)
        pipeline = Sam3Pipeline(bundle, device=device, cache_dir=cache)
        instance = cls(config.name)
        instance._pipeline = pipeline
        count, height, width = bundle.manifest["output_shape"]
        instance._metadata = Metadata(
            name="SAM3", type="instance_segmentation",
            description=f"SAM3 OpenVINO {device}; all proposals; native binary masks; CPU preprocessing/geometry glue",
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
        if self._active is not None:
            raise RuntimeError("SAM3 is busy; no inference requests are queued")
        if self._pipeline is None:
            raise RuntimeError("SAM3 is not configured")
        image, prompt = decode_inputs(input_tensors)
        future = asyncio.wrap_future(self._executor.submit(self._pipeline.infer, image, prompt))
        self._active = future

        def finished(completed: asyncio.Future) -> None:
            if self._active is completed:
                self._active = None
            if not completed.cancelled():
                completed.exception()

        future.add_done_callback(finished)
        return await asyncio.shield(future)

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
        active = self._active
        if active is not None:
            try:
                await asyncio.shield(active)
            except Exception:
                pass
        try:
            if self._pipeline is not None:
                await asyncio.get_running_loop().run_in_executor(self._executor, self._pipeline.close)
        finally:
            self._pipeline = None
            await asyncio.to_thread(self._executor.shutdown, wait=True)
