"""Real, small IR bundles exercise publication and the notebook compression scope."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import openvino as ov
import openvino.opset13 as ops
import pytest

from models.sam3_bundle import HF_REVISION, MODEL_NAMES, NOTEBOOK_REVISION, SAM3_REVISION, load_bundle


def compressor():
    path = Path(__file__).resolve().parents[1] / "scripts/compress_sam3.py"
    assert path.is_file(), "compression CLI is not implemented"
    spec = importlib.util.spec_from_file_location("compress_sam3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(root):
    return {str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def canonical_bundle(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    f = [1, 256, 1, 1]
    s = [1, 1, 256]
    p = [43, 1, 256]
    b = ([1, 43], ov.Type.boolean)
    contracts = {
        "image_encoder": ([[1, 3, 8, 8]], [f, f, f, f, f, f]),
        "text_encoder": ([([1, 32], ov.Type.i64)], [[32, 1, 256], ([1, 32], ov.Type.boolean)]),
        "geo_projections": ([[10, 1, 4], [10, 1, 4], [10, 256, 1, 1], ([10, 1], ov.Type.i64)], [[10, 1, 256]]),
        "geo_cross_attn": ([[10, 1, 256], ([1, 10], ov.Type.boolean), s, s], [[11, 1, 256], ([1, 11], ov.Type.boolean)]),
        "transformer_encoder": ([s, s, p, b], [s, s, [1], [1], [1], [1]]),
        "decoder": ([s, s, [1], [1], [1], [1], p, b], [[1, 3, 1, 256], [1, 3, 1, 4], [1, 1, 1]]),
        "scoring": ([[1, 1, 3, 256], [1, 1, 3, 4], p, b], [[1, 1, 3, 1], [1, 1, 1]]),
        "segmentation_head": ([f, f, f, [1, 3, 256], s, p, b], [[1, 3, 8, 8]]),
    }
    interfaces = {}
    rng = np.random.default_rng(7)
    for name, (inputs, outputs) in contracts.items():
        def shape_type(item):
            return item if isinstance(item, tuple) else (item, ov.Type.f32)
        params = [ops.parameter(*shape_type(item), name=f"input{i}") for i, item in enumerate(inputs)]
        scalar = ops.reduce_sum(ops.convert(params[0], ov.Type.f32), list(range(len(params[0].shape))), False)
        vector = ops.broadcast(scalar, [1, 128])
        product = vector
        for _ in range(3):
            weights = ops.constant(rng.standard_normal((128, 128)).astype(np.float32))
            product = ops.matmul(product, weights, False, False)
        value = ops.reduce_sum(product, [0, 1], False)
        result = []
        for i, item in enumerate(outputs):
            shape, dtype = shape_type(item)
            node = ops.broadcast(ops.convert(value, dtype), shape)
            node.output(0).get_tensor().set_names({f"output{i}"})
            result.append(node)
        model = ov.Model(result, params, name)
        ov.save_model(model, root / f"{name}.xml", compress_to_fp16=False)
        interfaces[name] = {direction: [{"name": port.get_any_name(), "shape": list(port.shape),
                                        "dtype": port.element_type.get_type_name()} for port in ports]
                            for direction, ports in (("inputs", model.inputs), ("outputs", model.outputs))}
    (root / "geo_config.json").write_bytes(b"{}")
    (root / "tokenizer.gz").write_bytes(b"vocabulary")
    (root / "extras").mkdir()
    (root / "extras/license").write_bytes(b"preserve me byte for byte\n")
    manifest = {
        "schema_version": 1, "adapter_version": 1, "model_id": "facebook/sam3",
        "checkpoint_revision": HF_REVISION, "checkpoint_sha256": "b" * 64,
        "sam3_revision": SAM3_REVISION, "notebook_revision": NOTEBOOK_REVISION,
        "precision": "FP32", "image_size": [8, 8], "context_length": 32, "output_shape": [3, 8, 8],
        "models": {name: f"{name}.xml" for name in MODEL_NAMES}, "interfaces": interfaces,
        "geo_config": "geo_config.json", "tokenizer": "tokenizer.gz",
        "preprocess": {"resize": "bilinear", "antialias": True, "mean": [0.5] * 3, "std": [0.5] * 3},
        "mask_semantics": "native_logits_gt_zero", "score_semantics": "matching_sigmoid_times_presence_sigmoid",
        "versions": {"openvino": ov.__version__, "torch": "2.9.1"}, "files": snapshot(root),
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    load_bundle(str(root))
    return root


def test_real_notebook_recipe_preserves_source_interfaces_and_other_files(canonical_bundle, tmp_path):
    nncf = pytest.importorskip("nncf")
    module = compressor()
    before = snapshot(canonical_bundle)
    target = tmp_path / "candidate"
    module.compress(canonical_bundle, target)
    assert snapshot(canonical_bundle) == before
    candidate = load_bundle(str(target))
    manifest = candidate.manifest
    original = json.loads((canonical_bundle / "manifest.json").read_text())
    assert manifest["precision"] == "MIXED"
    int4 = {"mode": "INT4_SYM", "group_size": 128, "ratio": 1.0,
            "group_size_fallback_mode": "IGNORE", "all_layers": False, "backup_mode": "INT8_ASYM"}
    assert manifest["compression"] == {
        "method": "nncf.compress_weights", "recipe": "notebook_sam3_weight_only",
        "nncf_version": nncf.__version__, "source_manifest_sha256": before["manifest.json"],
        "models": {"image_encoder": int4, "text_encoder": int4,
                   "transformer_encoder": {"mode": "INT8_ASYM"}, "decoder": {"mode": "INT8_ASYM"},
                   "segmentation_head": {"mode": "INT8_ASYM"}},
        "serialization": {"compress_to_fp16": True},
    }
    for key in original.keys() - {"precision", "files"}:
        assert manifest[key] == original[key]
    changed = {"manifest.json"} | {f"{name}.{suffix}" for name in (
        "image_encoder", "text_encoder", "transformer_encoder", "decoder", "segmentation_head")
        for suffix in ("xml", "bin")}
    after = snapshot(target)
    assert after.keys() == before.keys()
    for relative in before.keys() - changed:
        assert after[relative] == before[relative]
    for name in manifest["compression"]["models"]:
        assert (target / f"{name}.bin").stat().st_size < (canonical_bundle / f"{name}.bin").stat().st_size
        model = ov.Core().read_model(target / f"{name}.xml")
        types = {op.get_output_element_type(0).get_type_name() for op in model.get_ops() if op.get_type_name() == "Constant"}
        assert ("i4" in types) if name in ("image_encoder", "text_encoder") else ("u8" in types or "i8" in types)
        info = lambda key: model.get_rt_info(["nncf", "weight_compression", key]).value
        assert info("mode") == ("int4_sym" if name in ("image_encoder", "text_encoder") else "int8_asym")
        if name in ("image_encoder", "text_encoder"):
            assert info("group_size") == "128"
            assert info("ratio") == "1.0"
            assert info("all_layers") == "False"
            assert info("backup_mode") == "int8_asym"
    with pytest.raises(ValueError, match="FP32|compress"):
        module.compress(target, tmp_path / "again")
    assert not (tmp_path / "again").exists()


def test_existing_target_is_not_replaced(canonical_bundle, tmp_path):
    module = compressor()
    target = tmp_path / "candidate"
    target.mkdir()
    (target / "sentinel").write_bytes(b"untouched")
    with pytest.raises(ValueError, match="exist"):
        module.compress(canonical_bundle, target)
    assert snapshot(target) == {"sentinel": hashlib.sha256(b"untouched").hexdigest()}


def test_compression_failure_cleans_stage_without_publication(canonical_bundle, tmp_path, monkeypatch):
    nncf = pytest.importorskip("nncf")
    module = compressor()
    before = snapshot(canonical_bundle)
    def fail(*args, **kwargs):
        raise RuntimeError("compression failed")
    monkeypatch.setattr(nncf, "compress_weights", fail)
    with pytest.raises(RuntimeError, match="compression failed"):
        module.compress(canonical_bundle, tmp_path / "candidate")
    assert snapshot(canonical_bundle) == before
    assert {p.name for p in tmp_path.iterdir()} == {"source"}


@pytest.mark.parametrize("metadata", [None, {}, {"method": "nncf.compress_weights"}])
def test_mixed_bundle_requires_complete_recipe(canonical_bundle, metadata):
    path = canonical_bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["precision"] = "MIXED"
    manifest["compression"] = metadata
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="compression"):
        load_bundle(str(canonical_bundle))


def test_fp32_cannot_claim_compression(canonical_bundle):
    path = canonical_bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["compression"] = {}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="compression"):
        load_bundle(str(canonical_bundle))


def test_publication_does_not_clobber_concurrent_empty_directory(tmp_path):
    module = compressor()
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "manifest.json").write_bytes(b"complete")
    target = tmp_path / "candidate"
    target.mkdir()
    with pytest.raises(FileExistsError):
        module._publish(stage, target)
    assert list(target.iterdir()) == []
    assert (stage / "manifest.json").read_bytes() == b"complete"


def test_invalid_candidate_is_not_published(canonical_bundle, tmp_path, monkeypatch):
    pytest.importorskip("nncf")
    module = compressor()
    original_load = module.load_bundle
    def reject_candidate(path):
        if Path(path) != canonical_bundle:
            raise ValueError("candidate actual IR validation failed")
        return original_load(path)
    monkeypatch.setattr(module, "load_bundle", reject_candidate)
    with pytest.raises(ValueError, match="actual IR"):
        module.compress(canonical_bundle, tmp_path / "candidate")
    assert {p.name for p in tmp_path.iterdir()} == {"source"}


def test_nested_output_rejected_without_touching_source(canonical_bundle):
    module = compressor()
    before = snapshot(canonical_bundle)
    with pytest.raises(ValueError, match="outside"):
        module.compress(canonical_bundle, canonical_bundle / "candidate")
    assert snapshot(canonical_bundle) == before


@pytest.mark.parametrize("field,value", [
    ("method", "nncf.quantize"), ("recipe", "encoder_int8"),
    ("models", {"image_encoder": {"mode": "INT8_ASYM"}}),
    ("nncf_version", ""), ("nncf_version", None),
    ("source_manifest_sha256", "x" * 64), ("source_manifest_sha256", None),
    ("serialization", {"compress_to_fp16": False}), ("serialization", {"compress_to_fp16": 1}),
])
def test_mixed_recipe_rejects_inconsistent_metadata(canonical_bundle, field, value):
    path = canonical_bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    int4 = {"mode": "INT4_SYM", "group_size": 128, "ratio": 1.0, "group_size_fallback_mode": "IGNORE",
            "all_layers": False, "backup_mode": "INT8_ASYM"}
    manifest["precision"] = "MIXED"
    manifest["compression"] = {
        "method": "nncf.compress_weights", "recipe": "notebook_sam3_weight_only",
        "nncf_version": "3.4.0", "source_manifest_sha256": "a" * 64,
        "models": {"image_encoder": int4, "text_encoder": int4,
                   "transformer_encoder": {"mode": "INT8_ASYM"}, "decoder": {"mode": "INT8_ASYM"},
                   "segmentation_head": {"mode": "INT8_ASYM"}},
        "serialization": {"compress_to_fp16": True},
    }
    manifest["compression"][field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="compression"):
        load_bundle(str(canonical_bundle))


def test_output_parent_symlink_cannot_stage_inside_source(canonical_bundle, tmp_path, monkeypatch):
    module = compressor()
    alias = tmp_path / "alias"
    alias.symlink_to(canonical_bundle, target_is_directory=True)
    def never_stage(*args, **kwargs):
        raise AssertionError("source boundary must be checked before creating a staging directory")
    monkeypatch.setattr(module.tempfile, "TemporaryDirectory", never_stage)
    with pytest.raises(ValueError, match="outside"):
        module.compress(canonical_bundle, alias / "candidate")


def quantizer():
    path = Path(__file__).resolve().parents[1] / "scripts/quantize_sam3_image_encoder.py"
    spec = importlib.util.spec_from_file_location("quantize_sam3_image_encoder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ptq_metadata(**overrides):
    metadata = {
        "method": "nncf.quantize", "recipe": "image_encoder_int8_ptq", "nncf_version": "3.4.0",
        "source_manifest_sha256": "a" * 64, "calibration_samples": 4,
        "models": {"image_encoder": {
            "mode": "INT8", "model_type": "TRANSFORMER", "quantized": "weight_matmul",
            "ignored_types": ["Convolution", "ConvolutionBackpropData", "ScaledDotProductAttention"],
        }},
    }
    metadata.update(overrides)
    return metadata


def write_mixed(bundle, compression):
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["precision"] = "MIXED"
    manifest["compression"] = compression
    path.write_text(json.dumps(manifest))


def test_image_encoder_int8_manifest_loads(canonical_bundle):
    write_mixed(canonical_bundle, ptq_metadata())
    assert load_bundle(str(canonical_bundle)).manifest["compression"]["recipe"] == "image_encoder_int8_ptq"


@pytest.mark.parametrize("overrides", [
    {"method": "nncf.compress_weights"}, {"models": {"image_encoder": {"mode": "INT8"}}},
    {"calibration_samples": 0}, {"calibration_samples": True}, {"calibration_samples": None},
    {"nncf_version": ""}, {"source_manifest_sha256": "x" * 64}, {"serialization": {"compress_to_fp16": True}},
])
def test_image_encoder_int8_rejects_inconsistent_metadata(canonical_bundle, overrides):
    write_mixed(canonical_bundle, ptq_metadata(**overrides))
    with pytest.raises(ValueError, match="compression"):
        load_bundle(str(canonical_bundle))


def test_image_encoder_int8_changes_only_the_encoder(canonical_bundle, tmp_path):
    nncf = pytest.importorskip("nncf")
    pytest.importorskip("torchvision")
    module = quantizer()
    frame = tmp_path / "frame.npy"
    np.save(frame, np.random.default_rng(3).integers(0, 256, (12, 16, 3), dtype=np.uint8))
    before = snapshot(canonical_bundle)
    target = tmp_path / "int8"
    module.quantize(canonical_bundle, [frame], 3, target)
    assert snapshot(canonical_bundle) == before
    manifest = load_bundle(str(target)).manifest
    assert manifest["precision"] == "MIXED"
    assert manifest["compression"] == ptq_metadata(
        nncf_version=nncf.__version__, source_manifest_sha256=before["manifest.json"], calibration_samples=3)
    after = snapshot(target)
    assert after.keys() == before.keys()
    for relative in before.keys() - {"manifest.json", "image_encoder.xml", "image_encoder.bin"}:
        assert after[relative] == before[relative]
    model = ov.Core().read_model(target / "image_encoder.xml")
    assert any(op.get_type_name() == "FakeQuantize" for op in model.get_ops())
    with pytest.raises(ValueError, match="FP32"):
        module.quantize(target, [frame], 1, tmp_path / "again")
    assert not (tmp_path / "again").exists()
