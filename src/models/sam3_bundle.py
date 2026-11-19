"""Versioned artifacts for the SAM3-specific OpenVINO pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

MODEL_NAMES = (
    "image_encoder", "text_encoder", "geo_projections", "geo_cross_attn",
    "transformer_encoder", "decoder", "scoring", "segmentation_head",
)
SAM3_REVISION = "31cb7e624e4fe95301b1be1d4e809111593d2e45"
NOTEBOOK_REVISION = "343266b9537c5bfb2aad86916fbe0f2c2569250f"
HF_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"

NOTEBOOK_COMPRESSION_MODELS = {
    "image_encoder": {
        "mode": "INT4_SYM", "group_size": 128, "ratio": 1.0, "group_size_fallback_mode": "IGNORE",
        "all_layers": False, "backup_mode": "INT8_ASYM",
    },
    "text_encoder": {
        "mode": "INT4_SYM", "group_size": 128, "ratio": 1.0, "group_size_fallback_mode": "IGNORE",
        "all_layers": False, "backup_mode": "INT8_ASYM",
    },
    "transformer_encoder": {"mode": "INT8_ASYM"},
    "decoder": {"mode": "INT8_ASYM"},
    "segmentation_head": {"mode": "INT8_ASYM"},
}

IMAGE_ENCODER_PTQ_RECIPE = "image_encoder_int8_ptq"
IMAGE_ENCODER_PTQ_MODELS = {
    "image_encoder": {
        "mode": "INT8", "model_type": "TRANSFORMER", "quantized": "weight_matmul",
        "ignored_types": ["Convolution", "ConvolutionBackpropData", "ScaledDotProductAttention"],
    },
}


def _validate_precision(manifest: Dict[str, Any]) -> None:
    precision = manifest.get("precision")
    if precision == "FP32":
        if "compression" in manifest:
            raise ValueError("FP32 artifacts cannot declare compression")
        return
    if precision != "MIXED":
        raise ValueError("adapter version 1 requires FP32 or MIXED artifacts with a supported compression recipe")
    compression = manifest.get("compression")
    if not isinstance(compression, dict):
        raise ValueError("MIXED artifacts require complete compression metadata")
    if compression.get("recipe") == IMAGE_ENCODER_PTQ_RECIPE:
        _validate_image_encoder_ptq(compression)
    else:
        _validate_notebook_compression(compression)
    if not isinstance(compression["nncf_version"], str) or not compression["nncf_version"].strip():
        raise ValueError("compression nncf_version must identify the compressor")
    source_hash = compression["source_manifest_sha256"]
    if not isinstance(source_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", source_hash):
        raise ValueError("compression source_manifest_sha256 must be a SHA-256 digest")


def _validate_notebook_compression(compression: Dict[str, Any]) -> None:
    required = {"method", "recipe", "models", "nncf_version", "source_manifest_sha256", "serialization"}
    if set(compression) != required:
        raise ValueError("MIXED artifacts require complete notebook compression metadata")
    if compression["method"] != "nncf.compress_weights" or compression["recipe"] != "notebook_sam3_weight_only":
        raise ValueError("unsupported compression method or recipe")
    if compression["models"] != NOTEBOOK_COMPRESSION_MODELS:
        raise ValueError("compression models must match the pinned notebook recipe")
    for name in ("image_encoder", "text_encoder"):
        settings = compression["models"][name]
        if (type(settings["group_size"]) is not int or type(settings["ratio"]) not in (int, float)
                or type(settings["all_layers"]) is not bool):
            raise ValueError("compression group size, ratio and all_layers must use notebook setting types")
    if compression["serialization"] != {"compress_to_fp16": True} or (
        type(compression["serialization"]["compress_to_fp16"]) is not bool
    ):
        raise ValueError("compression serialization must record notebook FP16 float-constant defaults")


def _validate_image_encoder_ptq(compression: Dict[str, Any]) -> None:
    # Activation+weight INT8 of the image encoder's weight MatMuls only; the other seven networks stay FP32.
    required = {"method", "recipe", "models", "nncf_version", "source_manifest_sha256", "calibration_samples"}
    if set(compression) != required:
        raise ValueError("MIXED artifacts require complete image-encoder INT8 compression metadata")
    if compression["method"] != "nncf.quantize":
        raise ValueError("unsupported compression method or recipe")
    if compression["models"] != IMAGE_ENCODER_PTQ_MODELS:
        raise ValueError("compression models must match the image-encoder INT8 recipe")
    samples = compression["calibration_samples"]
    if type(samples) is not int or samples <= 0:
        raise ValueError("compression calibration_samples must be a positive integer")


@dataclass(frozen=True)
class Sam3Bundle:
    manifest: Dict[str, Any]
    model_paths: Dict[str, Path]
    geo_config_path: Path
    tokenizer_path: Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("artifact path must be a nonempty relative path")
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError(f"artifact path escapes outside bundle: {relative}")
    if not path.is_file():
        raise ValueError(f"missing artifact: {relative}")
    return path


def _dimensions(manifest: Dict[str, Any], key: str, rank: int) -> None:
    value = manifest.get(key)
    if not isinstance(value, list) or len(value) != rank or any(type(d) is not int or d <= 0 for d in value):
        raise ValueError(f"{key} must contain {rank} positive integer dimensions")


def load_bundle(model_dir: str) -> Sam3Bundle:
    root = Path(model_dir).resolve()
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"cannot read bundle manifest at {root}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("bundle manifest must be an object")
    for key in ("schema_version", "adapter_version"):
        if type(manifest.get(key)) is not int or manifest[key] != 1:
            raise ValueError(f"unsupported {key}: {manifest.get(key)!r}")
    if manifest.get("model_id") != "facebook/sam3":
        raise ValueError("model_id must be the official facebook/sam3 model")
    for key in ("checkpoint_revision", "sam3_revision", "notebook_revision"):
        if not isinstance(manifest.get(key), str) or not re.fullmatch(r"[0-9a-f]{40}", manifest[key]):
            raise ValueError(f"{key} must be an immutable source revision")
    checkpoint_hash = manifest.get("checkpoint_sha256")
    if not isinstance(checkpoint_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", checkpoint_hash):
        raise ValueError("checkpoint_sha256 must be a SHA-256 digest")
    _validate_precision(manifest)
    for key, expected in (
        ("checkpoint_revision", HF_REVISION),
        ("sam3_revision", SAM3_REVISION),
        ("notebook_revision", NOTEBOOK_REVISION),
    ):
        if manifest[key] != expected:
            raise ValueError(f"{key} is not supported by adapter version 1")
    if type(manifest.get("context_length")) is not int or manifest["context_length"] != 32:
        raise ValueError("context_length must be 32 including start/end tokens")
    _dimensions(manifest, "image_size", 2)
    _dimensions(manifest, "output_shape", 3)

    models = manifest.get("models")
    if not isinstance(models, dict) or set(models) != set(MODEL_NAMES):
        raise ValueError("models must identify the eight required SAM3 networks")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("files must map artifact paths to SHA-256 checksums")
    required = set()
    for name, relative in models.items():
        if not isinstance(relative, str) or Path(relative).suffix != ".xml":
            raise ValueError(f"{name} must identify an OpenVINO .xml artifact")
        required.update((relative, str(Path(relative).with_suffix(".bin"))))
    for key in ("geo_config", "tokenizer"):
        relative = manifest.get(key)
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"{key} must identify an artifact")
        required.add(relative)
    missing = required - files.keys()
    if missing:
        raise ValueError(f"missing artifact checksums: {', '.join(sorted(missing))}")
    paths = {}
    for relative, expected_hash in files.items():
        path = _bundle_path(root, relative)
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise ValueError(f"invalid checksum for {relative}")
        if file_sha256(path) != expected_hash:
            raise ValueError(f"checksum mismatch for {relative}")
        paths[relative] = path
    bundle = Sam3Bundle(
        manifest=manifest,
        model_paths={name: paths[relative] for name, relative in models.items()},
        geo_config_path=paths[manifest["geo_config"]],
        tokenizer_path=paths[manifest["tokenizer"]],
    )
    _validate_runtime_contract(bundle)
    return bundle


def _validate_runtime_contract(bundle: Sam3Bundle) -> None:
    manifest = bundle.manifest
    expected = {
        "preprocess": {"resize": "bilinear", "antialias": True, "mean": [0.5] * 3, "std": [0.5] * 3},
        "mask_semantics": "native_logits_gt_zero",
        "score_semantics": "matching_sigmoid_times_presence_sigmoid",
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"incompatible or missing {key}")
    versions = manifest.get("versions")
    if not isinstance(versions, dict) or any(not isinstance(versions.get(k), str) for k in ("openvino", "torch")):
        raise ValueError("versions must identify the OpenVINO and PyTorch exporters")
    declared = manifest.get("interfaces")
    if not isinstance(declared, dict) or set(declared) != set(MODEL_NAMES):
        raise ValueError("interfaces must describe all eight networks")
    import openvino as ov

    core = ov.Core()
    actual = {}
    counts = {
        "image_encoder": (1, None), "text_encoder": (1, 2), "transformer_encoder": (4, 6),
        "decoder": (8, 3), "scoring": (4, 2), "segmentation_head": (7, 1),
        "geo_projections": (4, 1), "geo_cross_attn": (4, 2),
    }
    for name, path in bundle.model_paths.items():
        model = core.read_model(path)
        input_count, output_count = counts[name]
        if len(model.inputs) != input_count or (output_count is not None and len(model.outputs) != output_count):
            raise ValueError(f"{name} interface has incompatible input/output counts")
        if name == "image_encoder" and len(model.outputs) not in (6, 12):
            raise ValueError("image_encoder outputs must contain three feature/position levels")
        interface = {}
        for direction, ports in (("inputs", model.inputs), ("outputs", model.outputs)):
            interface[direction] = [
                {"name": p.get_any_name(), "shape": list(p.shape), "dtype": p.element_type.get_type_name()} for p in ports
            ]
        if declared[name] != interface:
            raise ValueError(f"{name} interfaces differ from the manifest")
        actual[name] = interface
        del model

    def port(name, direction, index):
        return actual[name][direction][index]

    def require(name, direction, index, shape, dtype="f32"):
        p = port(name, direction, index)
        if p["shape"] != list(shape) or p["dtype"] != dtype:
            raise ValueError(f"{name} {direction}[{index}] interface has incompatible shape/dtype")

    def connect(source, source_index, target, target_index):
        p = port(source, "outputs", source_index)
        require(target, "inputs", target_index, p["shape"], p["dtype"])

    h, w = manifest["image_size"]
    n, mh, mw = manifest["output_shape"]
    require("image_encoder", "inputs", 0, [1, 3, h, w])
    require("text_encoder", "inputs", 0, [1, 32], "i64")
    require("text_encoder", "outputs", 0, [32, 1, 256])
    require("text_encoder", "outputs", 1, [1, 32], "boolean")
    require("segmentation_head", "outputs", 0, [1, n, mh, mw])
    lowest = port("image_encoder", "outputs", 2)["shape"]
    if len(lowest) != 4 or lowest[:2] != [1, 256]:
        raise ValueError("image_encoder lowest feature interface is incompatible")
    feature_shape = [lowest[2] * lowest[3], 1, 256]
    for index in (0, 1):
        require("transformer_encoder", "inputs", index, feature_shape)
    require("geo_cross_attn", "outputs", 0, [11, 1, 256])
    require("geo_cross_attn", "outputs", 1, [1, 11], "boolean")
    for name, indexes in (("transformer_encoder", (2, 3)), ("decoder", (6, 7)),
                          ("scoring", (2, 3)), ("segmentation_head", (5, 6))):
        require(name, "inputs", indexes[0], [43, 1, 256])
        require(name, "inputs", indexes[1], [1, 43], "boolean")
    for source_index, target_index in ((0, 0), (1, 1), (3, 3), (4, 4), (5, 5)):
        connect("transformer_encoder", source_index, "decoder", target_index)
    for output_index in (0, 1):
        p = port("decoder", "outputs", output_index)
        shape = p["shape"]
        if len(shape) != 4 or shape[1] != n or shape[2] != 1:
            raise ValueError("decoder outputs have incompatible proposal interfaces")
        require("scoring", "inputs", output_index, [shape[0], 1, n, shape[3]], p["dtype"])
    layers = port("decoder", "outputs", 0)["shape"][0]
    require("decoder", "outputs", 2, [layers, 1, 1])
    require("scoring", "outputs", 0, [layers, 1, n, 1])
    for index in range(3):
        connect("image_encoder", index, "segmentation_head", index)
    require("segmentation_head", "inputs", 3, [1, n, 256])
    connect("transformer_encoder", 0, "segmentation_head", 4)
    connect("geo_projections", 0, "geo_cross_attn", 0)
    require("geo_cross_attn", "inputs", 1, [1, 10], "boolean")
    require("geo_cross_attn", "inputs", 2, feature_shape)
    require("geo_cross_attn", "inputs", 3, feature_shape)
