"""Build a SAM3 bundle whose image encoder is INT8 post-training quantized; the other seven networks stay FP32.

Install requirements-sam3-compress.txt in an isolated environment. nncf.quantize with ModelType.TRANSFORMER
(SmoothQuant, mixed preset) quantizes weights and activations of the encoder's weight MatMuls (qkv/proj/fc1/fc2).
Convolutions (patch embed, FPN neck), deconvolutions, fused SDPA attention and activation x activation MatMuls
stay float. Calibration frames use the runtime preprocessing; extra samples are seeded crop/flip/colour
augmentations of those frames. Agreement with FP32 must be checked separately before any rollout.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compress_sam3 import _publish, _snapshot  # noqa: E402
from models.sam3_bundle import (  # noqa: E402
    IMAGE_ENCODER_PTQ_MODELS, IMAGE_ENCODER_PTQ_RECIPE, file_sha256, load_bundle,
)
from models.sam3_pipeline import image_transform  # noqa: E402

# Ops that may sit between a Constant and a MatMul weight input.
_WEIGHT_PATH_OPS = {"Convert", "Reshape", "Transpose", "Unsqueeze", "Squeeze", "Multiply", "Subtract"}


def load_rgb(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        pixels = np.load(path, allow_pickle=False)
    else:
        from PIL import Image

        with Image.open(path) as image:
            pixels = np.array(image.convert("RGB"), dtype=np.uint8)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError(f"{path}: expected RGB uint8 [H,W,3]")
    return pixels


def _is_constant(node, depth=0) -> bool:
    if node.get_type_name() == "Constant":
        return True
    return (node.get_type_name() in _WEIGHT_PATH_OPS and depth < 6
            and all(_is_constant(value.get_node(), depth + 1) for value in node.input_values()))


def _ignored_scope(model, nncf):
    ops = model.get_ordered_ops()
    activation_matmuls = [op.get_friendly_name() for op in ops
                          if op.get_type_name() == "MatMul" and not _is_constant(op.input_value(1).get_node())]
    present = {op.get_type_name() for op in ops}
    # NNCF rejects ignored types absent from the graph.
    types = [t for t in IMAGE_ENCODER_PTQ_MODELS["image_encoder"]["ignored_types"] if t in present]
    return nncf.IgnoredScope(names=activation_matmuls, types=types)


def _calibration(frames: list[Path], samples: int, size, seed: int):
    import torch
    from torchvision.transforms import v2

    unique = {}
    for path in frames:
        unique.setdefault(hashlib.sha256(load_rgb(path).tobytes()).hexdigest(), path)
    paths = list(unique.values())
    if not paths:
        raise ValueError("no calibration frames")
    to_tensor = image_transform(size)
    # (position, frame, augment): every unique frame once, then seeded augmentations cycling through them.
    items = [(position, paths[position % len(paths)], position >= len(paths))
             for position in range(max(samples, len(paths)))]

    def sample(item):
        position, path, augment = item
        pixels = load_rgb(path)
        if augment:
            torch.manual_seed(seed * 100003 + position)
            height, width = pixels.shape[:2]
            pixels = v2.Compose([
                v2.ToImage(),
                v2.RandomResizedCrop((height, width), scale=(0.5, 1.0),
                                     ratio=(0.8 * width / height, 1.25 * width / height), antialias=True),
                v2.RandomHorizontalFlip(),
                v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
            ])(pixels).permute(1, 2, 0).numpy()
        return to_tensor(pixels).unsqueeze(0).numpy()

    return items, sample, paths


def quantize(model_dir: Path, frames: list[Path], samples: int, output: Path, seed: int = 0) -> None:
    source = Path(model_dir).resolve()
    output = Path(os.path.abspath(output))
    output = output.parent.resolve() / output.name
    if output.exists() or output.is_symlink():
        raise ValueError(f"output already exists; refusing to replace it: {output}")
    if source == output or source in output.parents:
        raise ValueError("output must be outside the source bundle")
    original = _snapshot(source)
    bundle = load_bundle(str(source))
    if bundle.manifest["precision"] != "FP32" or "compression" in bundle.manifest:
        raise ValueError("quantization requires an uncompressed FP32 source bundle")
    import nncf
    import openvino as ov

    if nncf.__version__ != "3.4.0" or ov.__version__.split("-")[0] != "2026.4.0":
        raise ValueError("use the isolated pinned nncf==3.4.0 and openvino==2026.4.0 compression environment")
    relative = Path(bundle.manifest["models"]["image_encoder"])
    changed = {"manifest.json", str(relative), str(relative.with_suffix(".bin"))}
    items, sample, paths = _calibration(frames, samples, bundle.manifest["image_size"], seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sam3-int8-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        shutil.copytree(source, stage)
        if _snapshot(stage) != original:
            raise ValueError("source bundle changed while copying")
        print(f"Quantizing image_encoder with {len(items)} samples from {len(paths)} frames", flush=True)
        # Reading the destination would mmap weights that save_model truncates.
        model = ov.Core().read_model(source / relative)
        quantized = nncf.quantize(model, nncf.Dataset(items, sample), model_type=nncf.ModelType.TRANSFORMER,
                                  subset_size=len(items), ignored_scope=_ignored_scope(model, nncf))
        ov.save_model(quantized, stage / relative, compress_to_fp16=False)
        del quantized, model
        manifest = copy.deepcopy(bundle.manifest)
        manifest["precision"] = "MIXED"
        manifest["compression"] = {
            "method": "nncf.quantize", "recipe": IMAGE_ENCODER_PTQ_RECIPE, "nncf_version": nncf.__version__,
            "source_manifest_sha256": original["manifest.json"], "models": copy.deepcopy(IMAGE_ENCODER_PTQ_MODELS),
            "calibration_samples": len(items),
        }
        for path in manifest["files"]:
            manifest["files"][path] = file_sha256(stage / path)
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        candidate = _snapshot(stage)
        if candidate.keys() != original.keys() or any(candidate[p] != original[p] for p in original.keys() - changed):
            raise ValueError("quantization changed files other than the image encoder and manifest")
        load_bundle(str(stage))
        if _snapshot(source) != original:
            raise ValueError("source bundle changed during quantization; refusing publication")
        _publish(stage, output)
    print(json.dumps({
        "bundle": str(output), "manifest_sha256": file_sha256(output / "manifest.json"),
        "source_manifest_sha256": original["manifest.json"], "calibration_samples": len(items),
        "calibration_frames": [str(p) for p in paths],
        "image_encoder_bytes": (output / relative.with_suffix(".bin")).stat().st_size,
    }, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="FP32 source bundle")
    parser.add_argument("--frames", type=Path, nargs="+", required=True, help="RGB calibration images (.png/.jpg/.npy)")
    parser.add_argument("--samples", type=int, default=32, help="calibration samples; extras are augmentations")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    quantize(args.model_dir, args.frames, args.samples, args.output, args.seed)


if __name__ == "__main__":
    main()
