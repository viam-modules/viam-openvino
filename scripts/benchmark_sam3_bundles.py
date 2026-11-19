#!/usr/bin/env python3
"""Paired saved-image SAM3 GPU experiment; never approves quality or changes models.

Each variant runs in a fresh sequential process with an empty OpenVINO cache_dir.
The report's sibling <report-stem>.artifacts directory retains frozen RGB inputs,
all native masks/scores, worker logs, and unfiltered query-index overlay sheets.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def safe_error(exc):
    text = f"{type(exc).__name__}: {exc}"
    for name in ("VIAM_API_KEY", "VIAM_API_KEY_ID"):
        value = os.environ.get(name)
        if value:
            text = text.replace(value, "[REDACTED]")
    return text


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_dataset(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list) or not data["cases"]:
        raise ValueError("dataset requires a nonempty cases list")
    cases, arrays = [], []
    for index, case in enumerate(data["cases"]):
        if not isinstance(case, dict) or not isinstance(case.get("image"), str) or not case["image"]:
            raise ValueError(f"case {index}: image must be a nonempty path")
        if not isinstance(case.get("prompt"), str) or not case["prompt"].strip():
            raise ValueError(f"case {index}: prompt must be nonempty text")
        digest = case.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError(f"case {index}: SHA256 must contain 64 hexadecimal digits")
        image_path = (path.parent / case["image"]).resolve()
        encoded = image_path.read_bytes()
        actual = hashlib.sha256(encoded).hexdigest()
        if actual != digest.lower():
            raise ValueError(f"case {index}: encoded image SHA256 mismatch: {image_path}")
        with Image.open(io.BytesIO(encoded)) as image:
            pixels = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        cases.append({"index": index, "image": str(image_path), "prompt": case["prompt"],
                      "sha256": actual, "encoded_bytes": len(encoded), "rgb_shape": list(pixels.shape),
                      "rgb_sha256": hashlib.sha256(pixels.tobytes()).hexdigest()})
        arrays.append(pixels)
    return cases, arrays


def validate_outputs(output):
    masks, scores = output["masks"], output["scores"]
    if masks.dtype != np.uint8 or masks.ndim != 3 or min(masks.shape) <= 0:
        raise ValueError("masks must be nonempty uint8 [queries,height,width]")
    if scores.dtype != np.float32 or scores.shape != (masks.shape[0],) or not np.isfinite(scores).all():
        raise ValueError("scores must be finite float32 [queries] aligned with masks")
    if not np.all((masks == 0) | (masks == 1)):
        raise ValueError("masks must be binary")


def mask_boundaries(masks):
    padded = np.pad(masks, ((0, 0), (1, 1), (1, 1)), constant_values=False)
    interior = (padded[:, :-2, 1:-1] & padded[:, 2:, 1:-1]
                & padded[:, 1:-1, :-2] & padded[:, 1:-1, 2:])
    return masks & ~interior


def compare_outputs(baseline, candidate):
    validate_outputs(baseline)
    validate_outputs(candidate)
    if baseline["masks"].shape != candidate["masks"].shape:
        raise ValueError("mask shape mismatch; query remapping is forbidden")
    left, right = baseline["masks"].astype(bool), candidate["masks"].astype(bool)
    axes = (1, 2)
    intersection = np.logical_and(left, right).sum(axis=axes)
    union = np.logical_or(left, right).sum(axis=axes)
    iou = np.divide(intersection, union, out=np.ones(len(union), dtype=np.float64), where=union != 0)
    disagreement = left != right
    delta = candidate["scores"].astype(np.float64) - baseline["scores"].astype(np.float64)
    return {"per_query_iou": iou.tolist(), "empty_union_iou": 1.0,
            "per_query_disagreement_fraction": disagreement.mean(axis=axes).tolist(),
            "per_query_boundary_disagreement_fraction": (mask_boundaries(left) != mask_boundaries(right)).mean(axis=axes).tolist(),
            "boundary_definition": "foreground pixels touching background in 4-neighborhood; outside image is background",
            "disagreement_fraction": float(disagreement.mean()),
            "per_query_score_delta": delta.tolist(), "score_delta_direction": "candidate minus baseline",
            "max_absolute_score_delta": float(np.abs(delta).max()),
            "mean_absolute_score_delta": float(np.abs(delta).mean())}


def sample_schedule(case_count, repeats, warmup):
    if case_count <= 0 or repeats <= 0 or warmup < 0:
        raise ValueError("positive case count/repeats and nonnegative warmup required")
    return [("cold", 0, 0)] + [
        (phase, repeat, case_index)
        for phase, count in (("warmup", warmup), ("warm", repeats))
        for repeat in range(count) for case_index in range(case_count)
    ]


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("timing samples must be finite, nonnegative, and nonempty")
    return {"count": int(values.size), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95))}


def warm_summary(samples):
    warm = [sample for sample in samples if sample["phase"] == "warm"]
    keys = set(warm[0]["stages_ms"]) if warm else set()
    if any(set(sample["stages_ms"]) != keys for sample in warm):
        raise ValueError("warm samples have inconsistent timing stages")
    return {"wall_ms": distribution([sample["wall_ms"] for sample in warm]),
            "stages_ms": {key: distribution([sample["stages_ms"][key] for sample in warm]) for key in sorted(keys)}}


def validate_pairing(baseline, candidate):
    if [{k: v for k, v in case.items() if k != "frozen_rgb"} for case in baseline["cases"]] != [
        {k: v for k, v in case.items() if k != "frozen_rgb"} for case in candidate["cases"]
    ]:
        raise ValueError("baseline/candidate case identity or order differs")
    keys = ("phase", "repeat", "case_index")
    if [tuple(s[k] for k in keys) for s in baseline["samples"]] != [tuple(s[k] for k in keys) for s in candidate["samples"]]:
        raise ValueError("baseline/candidate sample order differs")


def require_gpu_execution(name, devices):
    devices = [str(device) for device in devices]
    if not devices or any(not re.fullmatch(r"GPU(?:\.\d+)*", device) for device in devices):
        raise RuntimeError(f"{name}: GPU execution required, compiled EXECUTION_DEVICES={devices}")
    return devices


def save_outputs(path, output):
    validate_outputs(output)
    np.savez_compressed(path, masks=output["masks"], scores=output["scores"])
    return {"path": str(Path(path).resolve()), "sha256": sha256_file(path),
            "file_bytes": Path(path).stat().st_size,
            "raw_output_bytes": int(output["masks"].nbytes + output["scores"].nbytes),
            "shapes": {key: list(output[key].shape) for key in ("masks", "scores")},
            "dtypes": {key: str(output[key].dtype) for key in ("masks", "scores")}}


def save_overlay(path, image, output):
    validate_outputs(output)
    count = output["masks"].shape[0]
    columns, width, height = min(10, count), 128, 148
    sheet = Image.new("RGB", (columns * width, math.ceil(count / columns) * height), "white")
    thumbnail = Image.fromarray(image).resize((width, width), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(sheet)
    for index, (mask, score) in enumerate(zip(output["masks"], output["scores"])):
        color = ((index * 73 + 70) % 256, (index * 151 + 90) % 256, (index * 197 + 120) % 256)
        alpha = Image.fromarray(mask * np.uint8(110)).resize((width, width), Image.Resampling.NEAREST)
        tile = Image.composite(Image.new("RGB", thumbnail.size, color), thumbnail, alpha)
        x, y = index % columns * width, index // columns * height
        sheet.paste(tile, (x, y))
        draw.text((x + 2, y + width + 2), f"q{index:03d} {float(score):.6f}", fill="black")
    sheet.save(path)


def _bundle_sizes(bundle, root):
    return {"total_file_bytes": sum(path.stat().st_size for path in root.rglob("*") if path.is_file()),
            "network_pair_bytes": {name: path.stat().st_size + path.with_suffix(".bin").stat().st_size
                                   for name, path in bundle.model_paths.items()}}


def _worker(job_path):
    sys.path.insert(0, str(ROOT / "src"))
    from models.sam3_bundle import load_bundle, MODEL_NAMES
    from models.sam3_pipeline import Sam3Pipeline
    import openvino as ov
    import torch
    import torchvision

    job = json.loads(Path(job_path).read_text())
    out_dir = Path(job["out_dir"])
    metadata = {"status": "running", "cases": job["cases"], "samples": [], "device": "GPU",
                "account_uid": os.getuid(), "host": os.uname().nodename,
                "versions": {"python": sys.version, "openvino": ov.__version__,
                             "torch": torch.__version__, "torchvision": torchvision.__version__}}
    pipeline = None
    try:
        arrays = []
        for case in job["cases"]:
            pixels = np.load(case["frozen_rgb"], allow_pickle=False)
            if (pixels.dtype != np.uint8 or list(pixels.shape) != case["rgb_shape"]
                    or hashlib.sha256(pixels.tobytes()).hexdigest() != case["rgb_sha256"]):
                raise ValueError("frozen RGB case identity mismatch")
            arrays.append(pixels)
        started = time.perf_counter()
        bundle = load_bundle(job["bundle_dir"])
        metadata["bundle_validation_ms"] = (time.perf_counter() - started) * 1000
        metadata["manifest_sha256"] = sha256_file(Path(job["bundle_dir"]) / "manifest.json")
        metadata["manifest"] = bundle.manifest
        metadata["sizes"] = _bundle_sizes(bundle, Path(job["bundle_dir"]))
        started = time.perf_counter()
        pipeline = Sam3Pipeline(bundle, device="GPU", cache_dir="")
        metadata["cold_compile_and_pipeline_init_ms"] = (time.perf_counter() - started) * 1000
        if set(pipeline._compiled) != set(MODEL_NAMES):
            raise RuntimeError("compiled network set differs from SAM3 contract")
        metadata["compiled_properties"] = {}
        for name, compiled in pipeline._compiled.items():
            devices = require_gpu_execution(name, compiled.get_property("EXECUTION_DEVICES"))
            properties = {"EXECUTION_DEVICES": devices}
            for key in ("INFERENCE_PRECISION_HINT", "PERFORMANCE_HINT", "NUM_STREAMS"):
                try:
                    properties[key] = str(compiled.get_property(key))
                except RuntimeError as exc:
                    properties[key] = {"unavailable": str(exc)}
            metadata["compiled_properties"][name] = properties
        for phase, repeat, case_index in sample_schedule(len(arrays), job["repeats"], job["warmup"]):
            stages = {}
            started = time.perf_counter()
            output = pipeline.infer(arrays[case_index], job["cases"][case_index]["prompt"], timings=stages)
            wall_ms = (time.perf_counter() - started) * 1000
            if set(stages) != set(MODEL_NAMES) | {"tokenization", "image_preprocess", "feature_layout", "geometry_cpu",
                                                "prompt_assembly", "query_layout", "postprocess", "pipeline_total"}:
                raise ValueError("request-local stage timings are incomplete")
            validate_outputs(output)
            if list(output["masks"].shape) != bundle.manifest["output_shape"]:
                raise ValueError("inference masks do not match validated bundle shape")
            stem = f"{phase}-{repeat:03d}-case-{case_index:03d}"
            saved = save_outputs(out_dir / f"{stem}.npz", output)
            metadata["samples"].append({"phase": phase, "repeat": repeat, "case_index": case_index,
                                        "wall_ms": wall_ms, "stages_ms": stages, "output": saved})
            if phase == "warm" and repeat == 0:
                overlay = out_dir / f"case-{case_index:03d}-all-queries.png"
                save_overlay(overlay, arrays[case_index], output)
                metadata.setdefault("overlays", []).append({"case_index": case_index, "path": str(overlay)})
        metadata["warm"] = warm_summary(metadata["samples"])
        metadata["warm_per_case"] = [warm_summary([s for s in metadata["samples"] if s["case_index"] == index])
                                     for index in range(len(arrays))]
        metadata["first_inference"] = metadata["samples"][0]
        metadata["status"] = "ok"
    except BaseException as exc:
        metadata.update(status="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        if pipeline is not None:
            pipeline.close()
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        metadata["process_peak_rss_bytes"] = int(rss if sys.platform == "darwin" else rss * 1024)
        metadata["memory_limitations"] = "Process lifetime high-water RSS includes load, inputs, outputs, overlays; not GPU/driver memory."
        write_json(out_dir / "variant.json", metadata)


def run_worker(job_path, timeout):
    log_path = Path(job_path).with_suffix(".log")
    with log_path.open("wb") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker-job", str(job_path)],
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
            raise
    if code:
        raise RuntimeError(f"variant process failed ({code}); see {log_path}")


def build_comparisons(baseline, candidate):
    validate_pairing(baseline, candidate)
    comparisons = []
    for left, right in zip(baseline["samples"], candidate["samples"]):
        for record in (left["output"], right["output"]):
            if sha256_file(record["path"]) != record["sha256"]:
                raise ValueError("saved output SHA256 mismatch")
        with np.load(left["output"]["path"], allow_pickle=False) as expected, np.load(right["output"]["path"], allow_pickle=False) as actual:
            metrics = compare_outputs(expected, actual)
        comparisons.append({**{key: left[key] for key in ("phase", "repeat", "case_index")}, **metrics,
                            "baseline_wall_ms": left["wall_ms"], "candidate_wall_ms": right["wall_ms"],
                            "wall_delta_ms": right["wall_ms"] - left["wall_ms"]})
    return comparisons


def validate_service_metadata(metadata):
    description = metadata.get("description", "")
    if not isinstance(description, str) or not description.startswith("SAM3 OpenVINO GPU;"):
        raise RuntimeError("service metadata must explicitly declare SAM3 OpenVINO GPU")
    return {"declared_device": "GPU", "description": description,
            "runtime_placement_proof": "service_metadata_and_explicit_device_config",
            "per_network_properties": "not_exposed_by_RPC",
            "actual_kernel_precision": "not_measured"}


async def run_rpc_variant(args, cases, artifacts, report):
    import asyncio
    import importlib.metadata
    from viam.robot.client import RobotClient
    from viam.rpc.dial import DialOptions
    from viam.services.mlmodel import MLModel
    from viam.services.mlmodel.utils import ndarrays_to_flat_tensors
    from google.protobuf.json_format import MessageToDict

    key_id, key = os.environ.get("VIAM_API_KEY_ID"), os.environ.get("VIAM_API_KEY")
    if not key_id or not key:
        raise ValueError("RPC mode requires VIAM_API_KEY_ID and VIAM_API_KEY environment variables")
    arrays = [np.load(case["frozen_rgb"], allow_pickle=False) for case in cases]
    for case, pixels in zip(cases, arrays):
        if hashlib.sha256(pixels.tobytes()).hexdigest() != case["rgb_sha256"]:
            raise ValueError("frozen RGB case identity mismatch")
    robot = await asyncio.wait_for(
        RobotClient.at_address(args.address, RobotClient.Options(
            dial_options=DialOptions.with_api_key(api_key=key, api_key_id=key_id))),
        timeout=min(args.timeout_seconds, 120),
    )
    try:
        for name, service_name, bundle_dir in [
            (args.variant, args.service, args.baseline_dir if args.variant == "baseline" else args.candidate_dir)
        ]:
            out_dir = artifacts / name
            out_dir.mkdir()
            service = MLModel.from_robot(robot, service_name)
            metadata = await service.metadata(timeout=min(args.infer_timeout, 60))
            service_metadata = MessageToDict(metadata)
            placement = validate_service_metadata(service_metadata)
            mask_shape = tuple(next(info.shape for info in metadata.output_info if info.name == "masks"))
            variant = {
                "status": "running", "cases": cases, "samples": [], "service": service_name,
                "host": os.uname().nodename, "account_uid": os.getuid(),
                "versions": {"python": sys.version, "viam-sdk": importlib.metadata.version("viam-sdk")},
                "measurement": "SDK infer round-trip including tensor codec, transport and server execution",
                "service_metadata": service_metadata, "gpu_placement": placement,
                "compiled_gpu_execution_verified": False,
                "cold_compile_and_pipeline_init_ms": None,
                "server_stage_timings": "not returned by MLModel infer; correlate separately with server DEBUG profiling logs",
                "memory_limitations": "Client RSS is not server/GPU memory; neither server RSS nor GPU allocations measured.",
            }
            if bundle_dir is not None:
                root = bundle_dir.resolve()
                variant["manifest_sha256"] = sha256_file(root / "manifest.json")
                variant["manifest"] = json.loads((root / "manifest.json").read_text())
                variant["sizes"] = {"total_file_bytes": sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
                                    "provenance": "local bundle directory; RPC does not attest server artifact hashes"}
            report["variants"][name] = variant
            try:
                for phase, repeat, case_index in sample_schedule(len(cases), args.repeats, args.warmup):
                    if phase == "cold":
                        phase = "initial"
                    inputs = {"image": arrays[case_index],
                              "prompt": np.frombuffer(cases[case_index]["prompt"].encode("utf-8"), dtype=np.uint8)}
                    started = time.perf_counter()
                    output = await service.infer(inputs, timeout=args.infer_timeout)
                    wall_ms = (time.perf_counter() - started) * 1000
                    validate_outputs(output)
                    if output["masks"].shape != mask_shape:
                        raise ValueError(f"RPC masks {output['masks'].shape} differ from service metadata {mask_shape}")
                    stem = f"{phase}-{repeat:03d}-case-{case_index:03d}"
                    saved = save_outputs(out_dir / f"{stem}.npz", output)
                    sample = {"phase": phase, "repeat": repeat, "case_index": case_index,
                              "wall_ms": wall_ms, "stages_ms": {}, "output": saved,
                              "raw_input_bytes": sum(value.nbytes for value in inputs.values()),
                              "tensor_payload_input_bytes": ndarrays_to_flat_tensors(inputs).ByteSize(),
                              "tensor_payload_output_bytes": ndarrays_to_flat_tensors(output).ByteSize()}
                    variant["samples"].append(sample)
                    if phase == "warm" and repeat == 0:
                        overlay = out_dir / f"case-{case_index:03d}-all-queries.png"
                        save_overlay(overlay, arrays[case_index], output)
                        variant.setdefault("overlays", []).append({"case_index": case_index, "path": str(overlay)})
                variant["first_observed_inference"] = variant["samples"][0]
                variant["warm"] = warm_summary(variant["samples"])
                variant["warm_per_case"] = [
                    warm_summary([s for s in variant["samples"] if s["case_index"] == index])
                    for index in range(len(cases))
                ]
                variant["status"] = "ok"
            except BaseException as exc:
                variant.update(status="error", error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                variant["client_process_peak_rss_bytes"] = int(rss if sys.platform == "darwin" else rss * 1024)
                write_json(out_dir / "variant.json", variant)
    finally:
        await asyncio.wait_for(robot.close(), timeout=15)


def compare_reports(baseline_report, candidate_report):
    if baseline_report["status"] != "ok" or candidate_report["status"] != "ok":
        raise ValueError("only completed reports may be compared")
    if baseline_report["dataset_sha256"] != candidate_report["dataset_sha256"]:
        raise ValueError("dataset manifest identity differs")
    for key in ("mode", "address", "service", "device", "repeats", "warmup", "infer_timeout_seconds"):
        if baseline_report["settings"].get(key) != candidate_report["settings"].get(key):
            raise ValueError(f"benchmark settings differ: {key}")
    baseline = baseline_report["variants"]["baseline"]
    candidate = candidate_report["variants"]["candidate"]
    if (baseline["host"], baseline["account_uid"], baseline["versions"]) != (
        candidate["host"], candidate["account_uid"], candidate["versions"]
    ):
        raise ValueError("variants did not use the same host/account/runtime")
    comparisons = build_comparisons(baseline, candidate)
    base_ms, cand_ms = (v["warm"]["wall_ms"]["median"] for v in (baseline, candidate))
    return {
        "schema_version": 1, "status": "ok", "settings": baseline_report["settings"],
        "dataset_sha256": baseline_report["dataset_sha256"], "cases": baseline_report["cases"],
        "dataset_coverage": baseline_report.get("dataset_coverage"),
        "variants": {"baseline": baseline, "candidate": candidate}, "comparisons": comparisons,
        "performance": {"warm_median_ratio_candidate_over_baseline": cand_ms / base_ms,
                        "interpretation": "observed before/after timings only; no quality or rollout approval"},
        "limitations": baseline_report["limitations"],
        "quality_verdict": "requires human review",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--address", help="Use live SDK RPC instead of isolated local bundle execution")
    parser.add_argument("--service", default="sam3-openvino")
    parser.add_argument("--variant", choices=["baseline", "candidate"])
    parser.add_argument("--compare-baseline", type=Path, help="Offline comparison: completed baseline report")
    parser.add_argument("--compare-candidate", type=Path, help="Offline comparison: completed candidate report")
    parser.add_argument("--infer-timeout", type=float, default=300, help="Per-RPC deadline in seconds")
    parser.add_argument("--device", choices=["GPU"], default="GPU")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=3600, help="Per-variant process deadline; kills isolated process group on timeout")
    parser.add_argument("--worker-job", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_job:
        _worker(args.worker_job)
        return 0
    if args.compare_baseline or args.compare_candidate:
        if not args.compare_baseline or not args.compare_candidate or not args.report:
            parser.error("offline comparison requires --compare-baseline, --compare-candidate and --report")
        if args.report.exists():
            parser.error("refusing to overwrite existing report")
        result = compare_reports(json.loads(args.compare_baseline.read_text()), json.loads(args.compare_candidate.read_text()))
        args.report.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.report, result)
        print(args.report.resolve())
        return 0
    required = ("dataset", "report") if args.address else ("baseline_dir", "candidate_dir", "dataset", "report")
    if any(getattr(args, name) is None for name in required):
        parser.error("--dataset/--report required; standalone also requires --baseline-dir/--candidate-dir")
    if args.repeats <= 0 or args.warmup < 0 or args.timeout_seconds <= 0 or not math.isfinite(args.infer_timeout) or args.infer_timeout <= 0:
        parser.error("repeats and timeouts must be positive; warmup must be nonnegative")
    if args.address and not args.variant:
        parser.error("RPC recording requires --variant baseline or --variant candidate")
    report_path = args.report.resolve()
    artifacts = report_path.with_name(report_path.stem + ".artifacts")
    if report_path.exists() or artifacts.exists():
        parser.error("refusing to overwrite existing report or artifacts")
    if args.baseline_dir and args.candidate_dir and args.baseline_dir.resolve() == args.candidate_dir.resolve():
        parser.error("baseline and candidate directories must differ")
    cases, arrays = load_dataset(args.dataset)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir()
    for index, (case, pixels) in enumerate(zip(cases, arrays)):
        path = artifacts / f"case-{index:03d}-rgb.npy"
        np.save(path, pixels, allow_pickle=False)
        case["frozen_rgb"] = str(path)
    del arrays
    report = {"schema_version": 1, "status": "running", "dataset": str(args.dataset.resolve()),
              "dataset_sha256": sha256_file(args.dataset), "cases": cases,
              "dataset_coverage": json.loads(args.dataset.read_text()).get("coverage"),
              "settings": {"device": "GPU", "repeats": args.repeats, "warmup": args.warmup,
                           "cache_dir": "", "execution_order": ["baseline", "candidate"],
                           "timeout_seconds_per_variant": args.timeout_seconds},
              "limitations": [
                  "Weight-only compression; inference precision hints are not proof of actual kernel precision.",
                  "EXECUTION_DEVICES proves selected GPU placement, not per-operation activation/kernel precision.",
                  "Cold compile timing includes pipeline/tokenizer initialization; filesystem and driver caches are not purged.",
                  "pipeline_total is inclusive; stage values are host wall milliseconds and must not be summed with it.",
                  "No RPC, camera acquisition or transport time included. Warm summaries exclude cold and warmup calls.",
                  "Process RSS is not GPU memory; GPU/driver allocations are not measured.",
                  "Sequential isolated variants do not establish absence of other GPU users; operator must ensure no contention.",
                  "Overlays show every query in original order at 128px square with nearest-neighbor mask display; outputs remain native.",
                  "No proposal remapping, confidence filtering, ground-truth accuracy measurement or automatic quality approval.",
                  "Baseline always runs first; repeatability/order/thermal effects require operator review or additional runs.",
              ], "variants": {}, "quality_verdict": "requires human review"}
    if args.address:
        report["settings"].update(mode="rpc", address=args.address, service=args.service,
                                  variant=args.variant, execution_order=[args.variant],
                                  infer_timeout_seconds=args.infer_timeout)
        report["settings"].update(cache_dir=None, timeout_seconds_total=args.timeout_seconds)
        del report["settings"]["timeout_seconds_per_variant"]
        report["limitations"] = [
            item for item in report["limitations"]
            if not item.startswith(("No RPC", "Cold compile", "Sequential isolated"))
        ] + [
            "RPC wall time includes SDK tensor encoding/decoding, network, server queueing and inference; no direct server/kernel timing.",
            "Initial RPC call is first observed, not proof of a cold service/model; compilation occurred outside this benchmark.",
            "GPU declaration checked via service metadata and explicit-device implementation; per-network properties not exposed by RPC.",
            "Tensor payload bytes are protobuf tensor sizes, not observed wire bytes; exclude framing, TLS, service names and metadata.",
            "Byte accounting and output persistence/overlays occur outside timed infer calls; warm summaries exclude initial/warmup calls.",
            "Calls are sequential on one connection; unrelated live-service activity/contention is not measured or controlled.",
        ]
    else:
        report["settings"]["mode"] = "standalone"
    try:
        if args.address:
            import asyncio
            asyncio.run(asyncio.wait_for(run_rpc_variant(args, cases, artifacts, report), timeout=args.timeout_seconds))
        else:
            for name, bundle_dir in (("baseline", args.baseline_dir), ("candidate", args.candidate_dir)):
                out_dir = artifacts / name
                out_dir.mkdir()
                job_path = artifacts / f"{name}-job.json"
                write_json(job_path, {"bundle_dir": str(bundle_dir.resolve()), "out_dir": str(out_dir),
                                      "cases": cases, "repeats": args.repeats, "warmup": args.warmup})
                run_worker(job_path, args.timeout_seconds)
                variant = json.loads((out_dir / "variant.json").read_text())
                if variant["status"] != "ok":
                    raise RuntimeError(f"{name} did not complete successfully")
                report["variants"][name] = variant
        report["status"] = "ok"
        if not args.address:
            report.update(compare_reports(report, report))
        report["status"] = "ok"
    except BaseException as exc:
        report.update(status="error", error=safe_error(exc))
        print(report["error"], file=sys.stderr)
        return 1
    finally:
        write_json(report_path, report)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
