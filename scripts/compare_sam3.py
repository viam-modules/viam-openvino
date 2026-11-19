"""Compare query-aligned native SAM3 outputs: Meta, pinned notebook, and adapter.

Runs each CPU implementation in a fresh, sequential process so original weights
and compiled OpenVINO models never coexist. No confidence filtering, proposal
reordering, or original-image mask interpolation participates in comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing
import resource
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.sam3_bundle import (
    HF_REVISION, NOTEBOOK_REVISION, SAM3_REVISION, file_sha256, load_bundle,
)

DEFAULT_IMAGE = (
    f"https://raw.githubusercontent.com/facebookresearch/sam3/{SAM3_REVISION}"
    "/assets/images/test_image.jpg"
)
# Fixed FP32 acceptance bounds, not fitted to this run. Boundary flips are reported.
TOLERANCES = {
    "preprocessed_image": (0.0, 0.0),
    "mask_logits": (5e-3, 1e-3),
    "matching_logits": (2e-3, 1e-3),
    "presence_logits": (2e-3, 1e-3),
    "scores": (2e-4, 1e-3),
}
MAX_BINARY_DISAGREEMENT = 1e-3


class _NativeOutputsReady(Exception):
    """Stop the notebook immediately after the real native mask-head execution."""


def _load_helper(path: Path):
    spec = importlib.util.spec_from_file_location("sam3_comparison_notebook", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import pinned notebook helper: {path}")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def _numpy(tensor):
    return tensor.detach().cpu().numpy().copy()


def _raw_result(masks, matching, presence, tokens, image):
    import numpy as np
    import torch

    masks, matching, presence = (np.array(value, copy=True) for value in (masks, matching, presence))
    if masks.ndim != 4 or masks.shape[0] != 1:
        raise ValueError(f"raw native masks are not [1,N,Hm,Wm]: {masks.shape}")
    count = masks.shape[1]
    if matching.shape != (1, count, 1) or presence.shape != (1, 1):
        raise ValueError(f"unexpected final raw logit axes: {matching.shape}, {presence.shape}")
    for value in (masks, matching, presence):
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("raw outputs must be finite CPU float32 logits")
    scores = (
        torch.from_numpy(matching).sigmoid()
        * torch.from_numpy(presence).sigmoid().unsqueeze(1)
    ).squeeze(-1)[0].numpy().copy()
    return {
        "mask_logits": masks,
        "matching_logits": matching,
        "presence_logits": presence,
        "scores": scores,
        "masks": (masks[0] > 0).astype(np.uint8),
        "tokens": np.array(tokens, copy=True),
        "preprocessed_image": np.array(image, copy=True),
    }


def _official(source, checkpoint, image, prompt, metadata):
    import torch
    from sam3 import model_builder
    from sam3.model.sam3_image_processor import Sam3Processor

    if Path(model_builder.__file__).resolve() != (source / "sam3/model_builder.py").resolve():
        raise RuntimeError("official model_builder was not imported from the pinned prepared source")
    start = time.perf_counter()
    torch.manual_seed(0)
    model = model_builder.build_sam3_image_model(
        bpe_path=str(source / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
        device="cpu", eval_mode=True, checkpoint_path=str(checkpoint),
        load_from_HF=False, enable_inst_interactivity=False, compile=False,
    ).float().eval()
    metadata["load_seconds"] = time.perf_counter() - start
    metadata["rope"] = "unchanged official complex RoPE; notebook patch_vit_rope not called"
    metadata["supervise_joint_box_scores"] = bool(model.supervise_joint_box_scores)
    processor = Sam3Processor(model, device="cpu", confidence_threshold=-1.0)
    captured = {}

    def capture_image(module, inputs):
        captured["image"] = _numpy(inputs[0])

    def capture_tokens(module, inputs):
        captured["tokens"] = _numpy(inputs[0])

    hooks = [
        model.backbone.vision_backbone.trunk.register_forward_pre_hook(capture_image),
        model.backbone.language_backbone.encoder.register_forward_pre_hook(capture_tokens),
    ]
    try:
        start = time.perf_counter()
        with torch.inference_mode():
            state = processor.set_image(image)
            metadata["image_encoder_seconds"] = time.perf_counter() - start
            state["backbone_out"].update(model.backbone.forward_text([prompt], device="cpu"))
            state["geometric_prompt"] = model._get_dummy_prompt()
            outputs = model.forward_grounding(
                backbone_out=state["backbone_out"], find_input=processor.find_stage,
                geometric_prompt=state["geometric_prompt"], find_target=None,
            )
            result = _raw_result(
                _numpy(outputs["pred_masks"]), _numpy(outputs["pred_logits"]),
                _numpy(outputs["presence_logit_dec"]), captured["tokens"], captured["image"],
            )
        metadata["inference_seconds"] = time.perf_counter() - start
        metadata["raw_output_path"] = "Sam3Processor.set_image + backbone.forward_text + model.forward_grounding"
        metadata["postprocessing"] = "processor filtering/interpolation never called; native tensors copied directly"
        return result
    finally:
        for hook in hooks:
            hook.remove()


def _notebook(source, helper_path, model_dir, image, prompt, metadata):
    import numpy as np
    import openvino as ov
    import torch
    from sam3.model.tokenizer_ve import SimpleTokenizer

    helper = _load_helper(helper_path)
    bundle = load_bundle(str(model_dir))
    core = ov.Core()
    captured = {}
    start = time.perf_counter()
    compiled = {
        name: core.compile_model(str(path), "CPU", {
            "PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f32",
        }) for name, path in bundle.model_paths.items()
    }
    metadata["compile_seconds"] = time.perf_counter() - start

    def observe(name):
        def invoke(inputs):
            if name in ("image_encoder", "text_encoder"):
                captured["image" if name == "image_encoder" else "tokens"] = np.array(inputs, copy=True)
            output = compiled[name](inputs)
            if name == "decoder":
                captured["presence"] = np.array(output[2], copy=True)
            elif name == "scoring":
                captured["matching"] = np.array(output[0], copy=True)
            elif name == "segmentation_head":
                captured["masks"] = np.array(output[0], copy=True)
                # The next upstream statements filter and interpolate; do neither.
                raise _NativeOutputsReady()
            return output
        return invoke

    processor = helper.OVSam3Processor(
        ov_image_encoder=observe("image_encoder"), ov_text_encoder=observe("text_encoder"),
        ov_transformer_encoder=observe("transformer_encoder"), ov_decoder=observe("decoder"),
        ov_scoring=observe("scoring"), ov_seg_head=observe("segmentation_head"),
        tokenizer=SimpleTokenizer(str(source / "sam3/assets/bpe_simple_vocab_16e6.txt.gz")),
        ov_geo_projections=observe("geo_projections"), ov_geo_cross_attn=observe("geo_cross_attn"),
        geo_config_path=str(bundle.geo_config_path), confidence_threshold=-1.0,
        resolution=bundle.manifest["image_size"][0],
    )
    start = time.perf_counter()
    with torch.inference_mode():
        state = processor.set_image(image)
        metadata["image_encoder_seconds"] = time.perf_counter() - start
        try:
            processor.set_text_prompt(prompt, state)
        except _NativeOutputsReady:
            pass
        else:
            raise RuntimeError("pinned notebook never reached the native mask-head capture")
    metadata["inference_seconds"] = time.perf_counter() - start
    metadata["raw_output_path"] = "unmodified pinned OVSam3Processor; intercepted real decoder/scoring/segmentation outputs"
    metadata["postprocessing"] = "stopped after segmentation execution, before upstream confidence filter or interpolation"
    return _raw_result(
        captured["masks"], captured["matching"][-1], captured["presence"][-1].T,
        captured["tokens"], captured["image"],
    )


def _adapter(model_dir, image, prompt, metadata):
    import numpy as np
    from models.sam3_pipeline import Sam3Pipeline, postprocess_proposals, tokenize_prompt

    bundle = load_bundle(str(model_dir))
    start = time.perf_counter()
    pipeline = Sam3Pipeline(bundle, device="CPU", cache_dir="")
    metadata["compile_seconds"] = time.perf_counter() - start
    try:
        pixels = np.array(image, dtype=np.uint8, copy=True)
        tokens = tokenize_prompt(pipeline._tokenizer, prompt).copy()
        transformed = pipeline._transform(pixels).unsqueeze(0).numpy().copy()
        start = time.perf_counter()
        # _forward returns InferRequest-backed views; own them before any next call.
        masks, matching, presence = (np.array(value, copy=True) for value in pipeline._forward(pixels, prompt))
        metadata["inference_seconds"] = time.perf_counter() - start
        result = _raw_result(masks, matching[-1], presence[-1].T, tokens, transformed)
        public = postprocess_proposals(masks, matching, presence)
        result["scores"] = public["scores"].copy()
        result["masks"] = public["masks"].copy()
        metadata["raw_output_path"] = "Sam3Pipeline._forward, copied before subsequent request use"
        metadata["postprocessing"] = "actual postprocess_proposals on captured native outputs; no second inference"
        return result
    finally:
        pipeline.close()


def _stage_worker(stage, source, helper, model_dir, checkpoint, image_path, prompt, output):
    metadata = {"stage": stage, "device": "CPU", "precision": "float32", "status": "running"}
    start = time.perf_counter()
    try:
        import numpy as np
        import openvino as ov
        import torch
        import torchvision
        from PIL import Image

        sys.path.insert(0, str(source))
        torch.set_grad_enabled(False)
        with Image.open(image_path) as original:
            image = original.convert("RGB")
        metadata["versions"] = {
            "python": sys.version, "torch": torch.__version__,
            "torchvision": torchvision.__version__, "openvino": ov.__version__,
        }
        if stage == "official":
            result = _official(source, checkpoint, image, prompt, metadata)
        elif stage == "notebook":
            result = _notebook(source, helper, model_dir, image, prompt, metadata)
        else:
            result = _adapter(model_dir, image, prompt, metadata)
        np.savez(output.with_suffix(".npz"), **result)
        metadata["shapes"] = {key: list(value.shape) for key, value in result.items()}
        metadata["dtypes"] = {key: str(value.dtype) for key, value in result.items()}
        metadata["status"] = "ok"
    except BaseException as exc:
        metadata.update(status="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        metadata["total_seconds"] = time.perf_counter() - start
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        metadata["peak_rss_bytes"] = int(rss if sys.platform == "darwin" else rss * 1024)
        output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


def _compare(reference_path, candidate_path):
    import numpy as np

    checks = {}
    with np.load(reference_path, allow_pickle=False) as reference, np.load(candidate_path, allow_pickle=False) as candidate:
        for name, (atol, rtol) in TOLERANCES.items():
            expected, actual = reference[name], candidate[name]
            check = {"atol": atol, "rtol": rtol, "reference_shape": list(expected.shape), "candidate_shape": list(actual.shape)}
            if expected.shape != actual.shape:
                check.update(passed=False, error="shape mismatch; no proposal remapping attempted")
            elif not np.isfinite(expected).all() or not np.isfinite(actual).all():
                check.update(passed=False, error="nonfinite output")
            else:
                delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
                close = delta <= atol + rtol * np.abs(expected)
                check.update(
                    passed=bool(close.all()), max_absolute_error=float(delta.max()),
                    mean_absolute_error=float(delta.mean()), outside_tolerance=int((~close).sum()),
                )
                if name == "mask_logits":
                    per_query = delta[0].reshape(delta.shape[1], -1)
                    check["per_query_max_absolute_error"] = per_query.max(axis=1).tolist()
                    check["per_query_mean_absolute_error"] = per_query.mean(axis=1).tolist()
                    check["failed_query_indices"] = np.flatnonzero(~close[0].reshape(delta.shape[1], -1).all(axis=1)).tolist()
            checks[name] = check
        expected, actual = reference["tokens"], candidate["tokens"]
        checks["tokenizer_ids"] = {
            "passed": bool(np.array_equal(expected, actual)),
            "reference": expected.tolist(), "candidate": actual.tolist(),
        }
        expected, actual = reference["masks"], candidate["masks"]
        binary = {"passed": False, "max_disagreement_fraction": MAX_BINARY_DISAGREEMENT}
        if expected.shape == actual.shape:
            disagreement = expected != actual
            binary.update(
                passed=bool(disagreement.mean() <= MAX_BINARY_DISAGREEMENT),
                disagreement_count=int(disagreement.sum()), disagreement_fraction=float(disagreement.mean()),
                per_query_disagreement_fraction=disagreement.reshape(expected.shape[0], -1).mean(axis=1).tolist(),
            )
        else:
            binary["error"] = "shape mismatch"
        checks["native_binary_masks"] = binary
    query_order = all(checks[name]["passed"] for name in ("mask_logits", "matching_logits", "scores"))
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "query_order": {"passed": query_order, "method": "same-index raw mask/matching/score comparisons for every query; no sorting, assignment, filtering or permutation"},
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="local real image path or HTTPS URL (default: pinned official example)")
    parser.add_argument("--prompt", default="shoe")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--stage-timeout", type=float, default=600.0, help="maximum seconds per isolated model stage")
    args = parser.parse_args()
    report = {
        "status": "running", "prompt": args.prompt,
        "model_dir": str(args.model_dir.resolve()), "checkpoint": str(args.checkpoint.resolve()),
        "image_source": args.image, "revisions": {
            "sam3": SAM3_REVISION, "notebook": NOTEBOOK_REVISION, "checkpoint": HF_REVISION,
        },
        "tolerance_policy": "fixed FP32 absolute + relative elementwise bounds; all raw logits must pass; binary threshold-boundary flips separately limited to 0.1%",
        "model_isolation": "sequential spawned processes, joined before starting the next implementation",
        "stages": {},
    }
    args.report = args.report.resolve()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        from PIL import Image
        from export_sam3 import prepare_source
        from models.sam3_pipeline import tokenize_prompt
        from models.sam3_reference import SimpleTokenizer

        bundle = load_bundle(str(args.model_dir))
        checkpoint = args.checkpoint.resolve(strict=True)
        checkpoint_hash = file_sha256(checkpoint)
        if checkpoint_hash != bundle.manifest["checkpoint_sha256"]:
            raise ValueError("comparison checkpoint does not match exported bundle checkpoint_sha256")
        if not args.prompt.strip():
            raise ValueError("prompt must contain non-whitespace text")
        tokenize_prompt(SimpleTokenizer(bundle.tokenizer_path), args.prompt)
        report["checkpoint_sha256"] = checkpoint_hash
        report["manifest_sha256"] = file_sha256(args.model_dir / "manifest.json")
        with tempfile.TemporaryDirectory(prefix="sam3-compare-") as temporary:
            work = Path(temporary)
            from check_sam3 import load_image

            pixels, image_info = load_image(args.image, work)
            report["image_file_sha256"] = image_info["sha256"]
            report["image_shape"] = list(pixels.shape)
            report["image_rgb_sha256"] = hashlib.sha256(pixels.tobytes()).hexdigest()
            canonical = work / "image.png"
            Image.fromarray(pixels).save(canonical)
            prepare_start = time.perf_counter()
            source, helper = prepare_source(work)
            report["source_preparation_seconds"] = time.perf_counter() - prepare_start
            report["cpu_source_patches"] = json.loads((work / "cpu_patches.json").read_text())
            report["notebook_helper_sha256"] = file_sha256(helper)
            report["baseline_differences"] = [
                "Only export_sam3.prepare_source CPU patches listed in cpu_source_patches modify Meta source.",
                "Original baseline retains official complex RoPE; only exported OpenVINO artifacts use the notebook's matrix replacement.",
                "All models use CPU float/eval without autocast or compilation of the original PyTorch model.",
                "Processor confidence_threshold=-1 preserves all scores, but comparison bypasses filtering and interpolation entirely.",
                "Notebook output capture stops with a private sentinel after its real segmentation call; raw logits remain in original query order.",
            ]
            context = multiprocessing.get_context("spawn")
            for stage in ("official", "notebook", "adapter"):
                print(f"Starting {stage} CPU comparison stage", flush=True)
                destination = work / stage
                process = context.Process(target=_stage_worker, args=(
                    stage, source, helper, args.model_dir.resolve(), checkpoint,
                    canonical, args.prompt, destination,
                ))
                process.start()
                try:
                    process.join(timeout=args.stage_timeout)
                    if process.is_alive():
                        raise TimeoutError(f"{stage} exceeded {args.stage_timeout}s")
                finally:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)
                        if process.is_alive():
                            process.kill()
                            process.join(timeout=5)
                stage_report = destination.with_suffix(".json")
                report["stages"][stage] = json.loads(stage_report.read_text()) if stage_report.exists() else {
                    "status": "error", "error": "worker exited without a report (possible out-of-memory termination)",
                }
                report["stages"][stage]["exitcode"] = process.exitcode
                if process.exitcode != 0:
                    raise RuntimeError(f"{stage} failed; independent native comparison unavailable: {report['stages'][stage].get('error')}")
                print(f"Finished {stage}: {report['stages'][stage]['total_seconds']:.2f}s", flush=True)
            report["comparisons"] = {
                f"{reference}_vs_{candidate}": _compare(work / f"{reference}.npz", work / f"{candidate}.npz")
                for reference, candidate in (("official", "notebook"), ("official", "adapter"), ("notebook", "adapter"))
            }
            report["status"] = "passed" if all(value["passed"] for value in report["comparisons"].values()) else "failed"
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        report["total_seconds"] = time.perf_counter() - start
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"SAM3 comparison {report['status']}; report: {args.report}", flush=True)
    if report["status"] != "passed":
        if "error" in report:
            print(report["error"], file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
