#!/usr/bin/env python3
"""Measure real SAM3 module RPCs without a robot, checkpoint, or export toolchain.

The JSON report is transport/contract evidence, not reference parity proof. All
outputs are saved, in query order, beside the report as an NPZ. An optional PNG
shows only the highest-scoring proposal, resized with nearest-neighbor sampling.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import sys
import tempfile
import time
import urllib.parse
import urllib.request

import numpy as np
from google.protobuf.json_format import MessageToDict
from grpclib.client import Channel
from grpclib.exceptions import GRPCError
from PIL import Image
from viam.proto.app.robot import ComponentConfig
from viam.proto.module import (
    AddResourceRequest,
    ModuleServiceStub,
    ReadyRequest,
    ReconfigureResourceRequest,
    RemoveResourceRequest,
)
from viam.proto.service.mlmodel import InferRequest, MetadataRequest, MLModelServiceStub
from viam.services.mlmodel.utils import flat_tensors_to_ndarrays, ndarrays_to_flat_tensors
from viam.utils import dict_to_struct


ROOT = Path(__file__).resolve().parents[1]
NAME = "sam3-check"
RESOURCE_NAME = "rdk:service:mlmodel/" + NAME
MAX_DOWNLOAD_BYTES = 32 * 1024 * 1024
DOWNLOAD_DEADLINE = 30.0


class HTTPSOnlyRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise ValueError("image redirects must remain HTTPS")
        return super().redirect_request(request, fp, code, message, headers, newurl)


def load_image(source: str, temporary: Path) -> tuple[np.ndarray, dict]:
    parsed = urllib.parse.urlsplit(source)
    path = Path(source).expanduser()
    downloaded = 0
    if parsed.scheme == "https":
        path = temporary / "downloaded-image"
        opener = urllib.request.build_opener(HTTPSOnlyRedirect())
        started = time.monotonic()
        request = urllib.request.Request(source, headers={"User-Agent": "sam3-rpc-check/1"})
        with opener.open(request, timeout=5) as response, path.open("wb") as target:
            length = response.headers.get("Content-Length")
            if length is not None and int(length) > MAX_DOWNLOAD_BYTES:
                raise ValueError("image download exceeds the 32 MiB limit")
            while True:
                if time.monotonic() - started > DOWNLOAD_DEADLINE:
                    raise TimeoutError("image download exceeded 30 seconds (plus at most one 5-second read)")
                chunk = response.read1(64 * 1024)
                if not chunk:
                    break
                downloaded += len(chunk)
                if downloaded > MAX_DOWNLOAD_BYTES:
                    raise ValueError("image download exceeds the 32 MiB limit")
                target.write(chunk)
    elif parsed.scheme:
        raise ValueError("--image must be a local path or an HTTPS URL")
    digest = hashlib.sha256()
    with path.open("rb") as encoded:
        for chunk in iter(lambda: encoded.read(1024 * 1024), b""):
            digest.update(chunk)
    with Image.open(path) as image:
        image.load()
        pixels = np.array(image.convert("RGB"), dtype=np.uint8)
    if pixels.size == 0:
        raise ValueError("image is empty")
    return pixels, {"source": source, "sha256": digest.hexdigest(), "shape": list(pixels.shape),
                    "downloaded_bytes": downloaded}


class MemorySampler:
    def __init__(self, pid: int):
        self.pid = pid
        self.phase = "process_startup"
        self.stopped = False
        self.phases: dict[str, dict] = {}
        self.errors: list[str] = []
        self.interval = 0.05
        try:
            import psutil
        except ImportError:
            self.psutil = None
        else:
            self.psutil = psutil

    async def run(self):
        if self.psutil is None:
            return
        previous = time.monotonic()
        root = self.psutil.Process(self.pid)
        while not self.stopped:
            now = time.monotonic()
            summary = self.phases.setdefault(self.phase, {
                "samples": 0, "max_observed_process_tree_rss_bytes": 0,
                "max_observed_root_rss_bytes": 0, "max_sampling_gap_seconds": 0.0,
            })
            try:
                root_rss = root.memory_info().rss
                total = root_rss
                for child in root.children(recursive=True):
                    with suppress(self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                        total += child.memory_info().rss
                summary["samples"] += 1
                summary["max_observed_process_tree_rss_bytes"] = max(
                    summary["max_observed_process_tree_rss_bytes"], total)
                summary["max_observed_root_rss_bytes"] = max(summary["max_observed_root_rss_bytes"], root_rss)
                summary["max_sampling_gap_seconds"] = max(summary["max_sampling_gap_seconds"], now - previous)
            except self.psutil.NoSuchProcess:
                return
            except self.psutil.AccessDenied as exc:
                if not self.errors:
                    self.errors.append(str(exc))
            previous = now
            await asyncio.sleep(self.interval)

    def report(self) -> dict:
        return {
            "status": "sampled" if any(p["samples"] for p in self.phases.values()) else "not_measured",
            "psutil_installed": self.psutil is not None,
            "requested_sampling_interval_seconds": self.interval,
            "phases": self.phases,
            "errors": self.errors,
            "limitations": "Observed RSS samples, not an OS high-water mark; short peaks and exited children can be missed. "
                           "Tree RSS can double-count shared pages. Excludes this checker and device VRAM. "
                           "Validation and compilation are not separately observable over AddResource RPC.",
        }


def config(model_dir: Path, device: str, cache: Path) -> ComponentConfig:
    return ComponentConfig(
        name=NAME, api="rdk:service:mlmodel", model="viam:openvino:sam3",
        attributes=dict_to_struct({"model_dir": str(model_dir), "device": device, "cache_dir": str(cache)}),
    )


def validate_metadata(response, shape: tuple[int, ...]) -> dict:
    metadata = response.metadata
    expected = {
        "input_info": [("image", "uint8", [-1, -1, 3]), ("prompt", "uint8", [-1])],
        "output_info": [("masks", "uint8", list(shape)), ("scores", "float32", [shape[0]])],
    }
    for field, entries in expected.items():
        actual = [(item.name, item.data_type, list(item.shape)) for item in getattr(metadata, field)]
        if actual != entries:
            raise AssertionError(f"metadata {field} differs: {actual!r}; expected {entries!r}")
    return MessageToDict(metadata, preserving_proto_field_name=True)


def validate_output(response, expected_shape: tuple[int, ...]) -> tuple[dict, dict]:
    values = flat_tensors_to_ndarrays(response.output_tensors)
    if set(values) != {"masks", "scores"}:
        raise AssertionError(f"unexpected output tensors: {sorted(values)}")
    masks, scores = values["masks"], values["scores"]
    if masks.dtype != np.uint8 or masks.shape != expected_shape:
        raise AssertionError(f"native masks must be uint8 {expected_shape}, got {masks.dtype} {masks.shape}")
    if scores.dtype != np.float32 or scores.shape != (expected_shape[0],):
        raise AssertionError(f"scores must be float32 [{expected_shape[0]}], got {scores.dtype} {scores.shape}")
    if not np.all((masks == 0) | (masks == 1)):
        raise AssertionError("masks contain nonbinary values")
    if not np.isfinite(scores).all() or not np.all((scores >= 0) & (scores <= 1)):
        raise AssertionError("scores must be finite probabilities in [0,1]")
    return values, {
        "status": "passed", "proposal_count": len(scores), "native_mask_shape": list(masks.shape),
        "masks_dtype": str(masks.dtype), "scores_dtype": str(scores.dtype),
        "scores_shape": list(scores.shape), "score_min": float(scores.min()), "score_max": float(scores.max()),
        "proposals_below_0_5": int(np.count_nonzero(scores < 0.5)),
        "empty_mask_count": int(np.count_nonzero(~masks.any(axis=(1, 2)))),
        "numpy_output_bytes": int(masks.nbytes + scores.nbytes),
        "output_tensors_protobuf_bytes": response.output_tensors.ByteSize(),
        "infer_response_protobuf_bytes": response.ByteSize(),
        "payload_measurement": "Serialized protobuf message sizes; excludes gRPC framing, HTTP/2, and transport overhead.",
        "ordering_limit": "Full manifest proposal count retained; query identity/order cannot be proven without a reference comparison.",
    }


def save_overlay(image: np.ndarray, values: dict, path: Path) -> dict:
    index = int(values["scores"].argmax())
    binary = Image.fromarray(values["masks"][index] * np.uint8(255)).resize(
        (image.shape[1], image.shape[0]), Image.Resampling.NEAREST)
    selected = np.asarray(binary) != 0
    overlay = image.copy()
    overlay[selected] = (0.55 * image[selected] + 0.45 * np.array([255, 30, 30])).astype(np.uint8)
    Image.fromarray(overlay).save(path)
    return {"path": str(path), "proposal_index": index, "score": float(values["scores"][index]),
            "resampling": "nearest; binary mask resized after inference", "human_inspection": "not_exercised"}


async def expect_still_waiting(awaitable, label: str, seconds: float = 2.0) -> dict:
    """A queued Infer must neither fail nor finish while the earlier worker still runs; it is abandoned after."""
    task = asyncio.ensure_future(awaitable)
    done, _ = await asyncio.wait({task}, timeout=seconds)
    if done:
        task.result()
        raise AssertionError(f"{label}: finished instead of waiting behind the running worker")
    task.cancel()
    with suppress(asyncio.CancelledError, GRPCError):
        await task
    return {"status": "passed", "waited_seconds": seconds}


async def expect_rpc_error(awaitable, label: str, text: str | None = None, status_name: str | None = None) -> dict:
    started = time.monotonic()
    try:
        await awaitable
    except GRPCError as exc:
        if text is not None and text not in str(exc).lower():
            raise AssertionError(f"{label}: expected error containing {text!r}, got {exc}") from exc
        if status_name is not None and exc.status.name != status_name:
            raise AssertionError(f"{label}: expected {status_name}, got {exc.status.name}") from exc
        return {"status": "passed", "error": str(exc), "grpc_status": exc.status.name,
                "seconds": time.monotonic() - started}
    raise AssertionError(f"{label}: RPC unexpectedly succeeded")


async def exercise_input_sequence(module, mlmodel, cfg, image, shape, args, temporary, report, sampler):
    result = report["additional_prompt_and_image_state_isolation"]
    result["status"] = "running"

    async def infer(pixels, prompt):
        request = InferRequest(name=NAME, input_tensors=ndarrays_to_flat_tensors({
            "image": pixels, "prompt": np.frombuffer(prompt.encode("utf-8"), dtype=np.uint8),
        }))
        response = await mlmodel.Infer(request, timeout=args.infer_timeout)
        return validate_output(response, shape)

    sampler.phase = "sequence_no_match_prompt"
    no_match, result["no_match_prompt"] = await infer(image, args.no_match_prompt)
    result["no_match_prompt"]["prompt"] = args.no_match_prompt
    np.savez(args.report.with_suffix(".no-match.npz"), **no_match)
    second_image, result["second_image"] = load_image(args.second_image, temporary)
    sampler.phase = "sequence_second_image"
    second, result["second_image_infer"] = await infer(second_image, args.prompt)
    np.savez(args.report.with_suffix(".second-image.npz"), **second)
    sampler.phase = "sequence_fresh_resource_reference"
    await module.ReconfigureResource(ReconfigureResourceRequest(config=cfg), timeout=args.startup_timeout)
    fresh, result["fresh_second_image_infer"] = await infer(second_image, args.prompt)
    np.testing.assert_allclose(second["scores"], fresh["scores"], atol=2e-4, rtol=1e-3)
    disagreement = float(np.mean(second["masks"] != fresh["masks"]))
    if disagreement > 1e-3:
        raise AssertionError(f"second-image masks differ from fresh-resource reference: {disagreement}")
    result["fresh_reference_comparison"] = {
        "status": "passed", "score_atol": 2e-4, "score_rtol": 1e-3,
        "score_max_absolute_error": float(np.max(np.abs(second["scores"] - fresh["scores"]))),
        "binary_disagreement_fraction": disagreement, "max_binary_disagreement_fraction": 1e-3,
        "method": "same-query comparison after changing both prompt and image within one resource lifetime",
    }
    result["status"] = "passed"


async def exercise_lifecycle(module, mlmodel, request, cfg, expected_shape, args, report, sampler):
    result = report["lifecycle"]
    result["status"] = "running"
    sampler.phase = "lifecycle_active_work"
    active = asyncio.create_task(mlmodel.Infer(request, timeout=args.infer_timeout))
    recreation = None
    try:
        await asyncio.sleep(0.25)
        if active.done():
            await active
            raise AssertionError("lifecycle inference finished before an active-work window was observed")
        result["concurrent_call_waits"] = await expect_still_waiting(
            mlmodel.Infer(request, timeout=args.infer_timeout), "concurrent Infer")
        active.cancel()
        with suppress(asyncio.CancelledError):
            await active
        result["caller_cancelled"] = True
        result["waits_after_caller_cancellation"] = await expect_still_waiting(
            mlmodel.Infer(request, timeout=args.infer_timeout), "Infer after cancellation")
        sampler.phase = "lifecycle_drain_and_warm_recreation"
        started = time.monotonic()
        recreation = asyncio.create_task(module.ReconfigureResource(
            ReconfigureResourceRequest(config=cfg), timeout=args.infer_timeout + args.startup_timeout))
        await asyncio.sleep(0.25)
        if recreation.done():
            await recreation
            raise AssertionError("reconfiguration completed before drain responsiveness could be observed")
        ready_started = time.monotonic()
        ready = await module.Ready(ReadyRequest(), timeout=5)
        if not ready.ready:
            raise AssertionError("module is not ready while draining")
        result["ready_during_drain_seconds"] = time.monotonic() - ready_started
        await recreation
        result["drain_and_warm_recreation_seconds"] = time.monotonic() - started
        result["metadata_after_recreation"] = validate_metadata(
            await mlmodel.Metadata(MetadataRequest(name=NAME), timeout=5), expected_shape)
        sampler.phase = "lifecycle_recreated_inference"
        response = await mlmodel.Infer(request, timeout=args.infer_timeout)
        _, result["infer_after_recreation"] = validate_output(response, expected_shape)
        invalid = ComponentConfig()
        invalid.CopyFrom(cfg)
        invalid.attributes.CopyFrom(dict_to_struct({"model_dir": "", "device": args.device, "cache_dir": ""}))
        sampler.phase = "lifecycle_failed_recreation"
        result["failed_recreation"] = await expect_rpc_error(module.ReconfigureResource(
            ReconfigureResourceRequest(config=invalid), timeout=args.startup_timeout), "invalid recreation", "model_dir")
        result["old_resource_absent_after_failure"] = await expect_rpc_error(
            mlmodel.Metadata(MetadataRequest(name=NAME), timeout=5), "metadata after failed recreation",
            status_name="NOT_FOUND")
        sampler.phase = "lifecycle_recovery_add"
        await module.AddResource(AddResourceRequest(config=cfg), timeout=args.startup_timeout)
        validate_metadata(await mlmodel.Metadata(MetadataRequest(name=NAME), timeout=5), expected_shape)
        await module.RemoveResource(RemoveResourceRequest(name=RESOURCE_NAME), timeout=args.shutdown_timeout)
        result["resource_absent_after_remove"] = await expect_rpc_error(
            mlmodel.Metadata(MetadataRequest(name=NAME), timeout=5), "metadata after RemoveResource",
            status_name="NOT_FOUND")
        result["limitations"] = (
            "External RPC evidence only; exact internal worker-release ordering requires the controlled SDK lifecycle tests. "
            "Active-work overlap is asserted; a run too fast to observe it fails rather than passing silently.")
        result["status"] = "passed"
    finally:
        for task in (active, recreation):
            if task is not None:
                if not task.done():
                    task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task


async def wait_for_socket(process, path: Path, timeout: float):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if process.returncode is not None:
            raise RuntimeError(f"module exited before creating its socket: {process.returncode}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"module socket did not appear within {timeout} seconds")
        await asyncio.sleep(0.05)


async def stop_process(process, timeout: float) -> dict:
    if process.returncode is not None:
        return {"returncode": process.returncode, "termination": "already_exited"}
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        await asyncio.wait_for(process.wait(), timeout)
        return {"returncode": process.returncode, "termination": "SIGTERM"}
    except asyncio.TimeoutError:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        await asyncio.wait_for(process.wait(), timeout)
        return {"returncode": process.returncode, "termination": "SIGKILL_after_shutdown_timeout"}


async def run(args, report: dict):
    process = None
    channel = None
    sampler = None
    sampling = None
    added = False
    with tempfile.TemporaryDirectory(prefix="s3-", dir="/tmp") as directory:
        temporary = Path(directory)
        image, report["image"] = load_image(args.image, temporary)
        manifest_bytes = (args.model_dir / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        shape = tuple(manifest["output_shape"])
        if len(shape) != 3 or any(type(value) is not int or value <= 0 for value in shape):
            raise ValueError("manifest output_shape must be three positive integers")
        report["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
        cache = temporary / "cold-cache"
        cache.mkdir()
        cfg = config(args.model_dir, args.device, cache)
        socket = temporary / "m.sock"
        entrypoint = args.module_entrypoint
        command = ([sys.executable, str(entrypoint)] if entrypoint.suffix == ".py" else [str(entrypoint)]) + [str(socket)]
        report["module_command"] = command
        report["module_kind"] = "source" if entrypoint.suffix == ".py" else "executable"
        environment = dict(os.environ, VIAM_NO_MODULE_PARENT="true", VIAM_MODULE_DATA=str(temporary / "module-data"))
        log_path = args.report.with_suffix(".module.log")
        report["module_log"] = str(log_path)
        with log_path.open("wb") as log:
            try:
                started = time.monotonic()
                process = await asyncio.create_subprocess_exec(
                    *command, cwd=ROOT, env=environment, stdout=log, stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True)
                sampler = MemorySampler(process.pid)
                sampling = asyncio.create_task(sampler.run())
                await wait_for_socket(process, socket, args.socket_timeout)
                channel = Channel(path=str(socket))
                module = ModuleServiceStub(channel)
                mlmodel = MLModelServiceStub(channel)
                ready = await module.Ready(ReadyRequest(), timeout=5)
                if not ready.ready or not any("viam:openvino:sam3" in h.models for h in ready.handlermap.handlers):
                    raise AssertionError("Ready did not advertise the SAM3 model")
                report["process_startup_seconds"] = time.monotonic() - started
                sampler.phase = "cold_add_resource"
                startup = report["cold_add_resource"] = {
                    "status": "running", "cache_initially_empty": not any(cache.iterdir()),
                    "configured_deadline_seconds": args.startup_timeout,
                    "deadline_note": "Checker RPC deadline, not a modification to any robot configuration. "
                                     "If insufficient, explicitly pass --startup-timeout and adjust the real server deadline separately.",
                }
                if not startup["cache_initially_empty"]:
                    raise AssertionError("cold cache is not empty")
                started = time.monotonic()
                try:
                    await module.AddResource(AddResourceRequest(config=cfg), timeout=args.startup_timeout)
                    added = True
                finally:
                    startup["seconds"] = time.monotonic() - started
                startup["within_deadline"] = startup["seconds"] <= args.startup_timeout
                if not startup["within_deadline"]:
                    raise TimeoutError("cold AddResource exceeded the configured deadline")
                startup["cache_files_after_configuration"] = sum(p.is_file() for p in cache.rglob("*"))
                startup["status"] = "passed"
                report["metadata"] = validate_metadata(
                    await mlmodel.Metadata(MetadataRequest(name=NAME), timeout=5), shape)
                sampler.phase = "infer"
                started = time.monotonic()
                request = InferRequest(name=NAME, input_tensors=ndarrays_to_flat_tensors({
                    "image": image, "prompt": np.frombuffer(args.prompt.encode("utf-8"), dtype=np.uint8),
                }))
                report["infer_request_protobuf_bytes"] = request.ByteSize()
                rpc_started = time.monotonic()
                response = await mlmodel.Infer(request, timeout=args.infer_timeout)
                roundtrip = time.monotonic() - rpc_started
                values, contract = validate_output(response, shape)
                contract["rpc_roundtrip_seconds"] = roundtrip
                contract["prepare_rpc_and_decode_seconds"] = time.monotonic() - started
                report["infer"] = contract
                npz_path = args.report.with_suffix(".npz")
                np.savez(npz_path, **values)
                report["outputs_npz"] = str(npz_path)
                if args.overlay:
                    report["overlay"] = save_overlay(image, values, args.report.with_suffix(".overlay.png"))
                if args.second_image:
                    await exercise_input_sequence(module, mlmodel, cfg, image, shape, args, temporary, report, sampler)
                if args.exercise_lifecycle:
                    await exercise_lifecycle(module, mlmodel, request, cfg, shape, args, report, sampler)
                    added = False
                report["status"] = "passed"
            finally:
                if sampler is not None:
                    sampler.phase = "shutdown"
                try:
                    if channel is not None and added:
                        try:
                            await ModuleServiceStub(channel).RemoveResource(
                                RemoveResourceRequest(name=RESOURCE_NAME), timeout=args.shutdown_timeout)
                            report["cleanup_remove_resource"] = "passed"
                        except Exception as exc:
                            report["cleanup_remove_resource"] = f"failed: {type(exc).__name__}: {exc}"
                            report["status"] = "failed"
                    if channel is not None:
                        channel.close()
                    if process is not None:
                        report["process_shutdown"] = await stop_process(process, args.shutdown_timeout)
                        if report["process_shutdown"]["termination"] == "SIGKILL_after_shutdown_timeout":
                            report["status"] = "failed"
                finally:
                    if sampler is not None:
                        sampler.stopped = True
                    if sampling is not None:
                        try:
                            await asyncio.wait_for(sampling, 2)
                        except Exception as exc:
                            report["memory_sampling_error"] = f"{type(exc).__name__}: {exc}"
                    if sampler is not None:
                        report["memory"] = sampler.report()


def positive_seconds(value: str) -> float:
    seconds = float(value)
    if not np.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("timeout must be a positive finite number of seconds")
    return seconds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--image", required=True, help="local image path or HTTPS URL (32 MiB maximum)")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--second-image", help="also verify no-match prompt and a second image in the same resource lifetime")
    parser.add_argument("--no-match-prompt", default="a transparent spaceship", help="absent-object prompt for the optional input sequence")
    parser.add_argument("--device", choices=("CPU", "GPU"), default="CPU")
    parser.add_argument("--module-entrypoint", type=Path, default=ROOT / "src/main.py")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--exercise-lifecycle", action="store_true", help="also run active/cancelled-work reconfiguration, failure, recovery, and removal RPCs")
    parser.add_argument("--overlay", action="store_true", help="save a resized binary overlay of the highest-scoring proposal")
    parser.add_argument("--startup-timeout", type=positive_seconds, default=120.0)
    parser.add_argument("--infer-timeout", type=positive_seconds, default=600.0)
    parser.add_argument("--socket-timeout", type=positive_seconds, default=30.0)
    parser.add_argument("--shutdown-timeout", type=positive_seconds, default=30.0)
    args = parser.parse_args()
    args.model_dir = args.model_dir.expanduser().resolve()
    args.module_entrypoint = args.module_entrypoint.expanduser().resolve()
    args.report = args.report.expanduser().resolve()
    if not args.prompt.strip():
        parser.error("--prompt must contain non-whitespace text")
    if args.report.suffix != ".json":
        parser.error("--report must end in .json")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1, "status": "running", "device_requested": args.device,
        "model_dir": str(args.model_dir), "prompt": args.prompt,
        "platform": platform.platform(), "python": sys.version,
        "timeouts_seconds": {"startup": args.startup_timeout, "infer": args.infer_timeout,
                             "socket": args.socket_timeout, "shutdown": args.shutdown_timeout},
        "lifecycle": {"status": "not_exercised" if not args.exercise_lifecycle else "pending"},
        "reference_parity": {"status": "not_exercised", "reason": "Use the independent reference checker; this tool never loads a checkpoint."},
        "additional_prompt_and_image_state_isolation": {"status": "pending" if args.second_image else "not_exercised",
                                                       "reason": "Pass --second-image to exercise and compare consecutive inputs."},
        "warm_add_resource": {"status": "not_exercised", "reason": "Optional lifecycle measures drain plus warm recreation, not isolated warm AddResource."},
    }
    try:
        asyncio.run(run(args, report))
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        for phase in ("cold_add_resource", "lifecycle", "additional_prompt_and_image_state_isolation"):
            if report.get(phase, {}).get("status") == "running":
                report[phase]["status"] = "failed"
    finally:
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{report['status']}: {args.report}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
