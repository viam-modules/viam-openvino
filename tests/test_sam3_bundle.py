import hashlib
import json
import shutil

import pytest

from models.sam3_bundle import load_bundle


@pytest.fixture
def bundle_dir(tmp_path, tiny_ir):
    root = tmp_path / "bundle"
    root.mkdir()
    files = {}
    names = (
        "image_encoder", "text_encoder", "geo_projections", "geo_cross_attn",
        "transformer_encoder", "decoder", "scoring", "segmentation_head",
    )
    for name in names:
        for suffix in (".xml", ".bin"):
            source = tiny_ir if suffix == ".xml" else tiny_ir.replace(".xml", ".bin")
            path = root / (name + suffix)
            shutil.copyfile(source, path)
            files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name, content in (("geo_config.json", b"{}"), ("bpe_simple_vocab_16e6.txt.gz", b"vocabulary")):
        (root / name).write_bytes(content)
        files[name] = hashlib.sha256(content).hexdigest()
    manifest = {
        "schema_version": 1,
        "adapter_version": 1,
        "model_id": "facebook/sam3",
        "checkpoint_revision": "3c879f39826c281e95690f02c7821c4de09afae7",
        "checkpoint_sha256": "b" * 64,
        "sam3_revision": "31cb7e624e4fe95301b1be1d4e809111593d2e45",
        "notebook_revision": "343266b9537c5bfb2aad86916fbe0f2c2569250f",
        "precision": "FP32",
        "image_size": [8, 8],
        "context_length": 32,
        "output_shape": [3, 8, 8],
        "models": {name: name + ".xml" for name in names},
        "files": files,
        "geo_config": "geo_config.json",
        "tokenizer": "bpe_simple_vocab_16e6.txt.gz",
        "preprocess": {"resize": "bilinear", "antialias": True, "mean": [0.5] * 3, "std": [0.5] * 3},
        "mask_semantics": "native_logits_gt_zero",
        "score_semantics": "matching_sigmoid_times_presence_sigmoid",
        "versions": {"openvino": "2026.4.0", "torch": "2.9.1"},
        "interfaces": {name: {"inputs": [{"name": "images", "shape": [1, 3, 8, 8], "dtype": "f32"}],
                              "outputs": [{"name": "output0", "shape": [1, 4, 8, 8], "dtype": "f32"}]}
                       for name in names},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def change_manifest(root, **changes):
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(changes)
    path.write_text(json.dumps(manifest))


def test_missing_weights_rejected_before_serving(bundle_dir):
    (bundle_dir / "decoder.bin").unlink()
    with pytest.raises(ValueError, match="decoder.bin"):
        load_bundle(str(bundle_dir))


def test_modified_artifact_rejected_before_serving(bundle_dir):
    with (bundle_dir / "decoder.bin").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(str(bundle_dir))


def test_incompatible_adapter_rejected(bundle_dir):
    change_manifest(bundle_dir, adapter_version=2)
    with pytest.raises(ValueError, match="adapter"):
        load_bundle(str(bundle_dir))


@pytest.mark.parametrize("field,value", [("image_size", [0, 8]), ("output_shape", [3, -1, 8]), ("context_length", 77)])
def test_incompatible_dimensions_rejected(bundle_dir, field, value):
    change_manifest(bundle_dir, **{field: value})
    with pytest.raises(ValueError, match=field):
        load_bundle(str(bundle_dir))


def test_path_escape_rejected_even_with_matching_hash(bundle_dir, tmp_path):
    external = tmp_path / "outside.gz"
    external.write_bytes(b"outside")
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    manifest["tokenizer"] = "../outside.gz"
    manifest["files"]["../outside.gz"] = hashlib.sha256(b"outside").hexdigest()
    change_manifest(bundle_dir, **manifest)
    with pytest.raises(ValueError, match="outside|escape|relative"):
        load_bundle(str(bundle_dir))


def test_required_artifact_cannot_be_omitted_from_checksums(bundle_dir):
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    del manifest["files"]["decoder.bin"]
    change_manifest(bundle_dir, **manifest)
    with pytest.raises(ValueError, match="decoder.bin"):
        load_bundle(str(bundle_dir))


@pytest.mark.parametrize("field", ["checkpoint_revision", "sam3_revision", "notebook_revision"])
def test_different_source_revision_requires_compatible_adapter(bundle_dir, field):
    change_manifest(bundle_dir, **{field: "e" * 40})
    with pytest.raises(ValueError, match=field):
        load_bundle(str(bundle_dir))


@pytest.mark.parametrize("field", ["preprocess", "interfaces", "versions", "mask_semantics", "score_semantics"])
def test_missing_runtime_contract_metadata_rejected(bundle_dir, field):
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    del manifest[field]
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=field):
        load_bundle(str(bundle_dir))


def test_unrelated_ir_cannot_pass_as_an_eight_network_pipeline(bundle_dir):
    with pytest.raises(ValueError, match="interface|inputs|outputs"):
        load_bundle(str(bundle_dir))
