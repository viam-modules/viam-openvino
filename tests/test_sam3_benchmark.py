"""Deterministic report/I/O tests; no simulated inference acceptance or GPU use."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

SPEC = importlib.util.spec_from_file_location(
    "benchmark_sam3_bundles", Path(__file__).resolve().parents[1] / "scripts/benchmark_sam3_bundles.py",
)
benchmark = importlib.util.module_from_spec(SPEC)


@pytest.fixture(scope="module", autouse=True)
def load_script():
    SPEC.loader.exec_module(benchmark)


def test_query_aligned_comparison_reports_empty_union_and_signed_score_deltas():
    baseline = {"masks": np.array([[[1, 0], [0, 0]], [[0, 0], [0, 0]]], dtype=np.uint8),
                "scores": np.array([0.5, 0.25], dtype=np.float32)}
    candidate = {"masks": np.array([[[1, 1], [0, 0]], [[0, 0], [0, 0]]], dtype=np.uint8),
                 "scores": np.array([0.75, 0.125], dtype=np.float32)}
    report = benchmark.compare_outputs(baseline, candidate)
    assert report["per_query_iou"] == [0.5, 1.0]
    assert report["per_query_disagreement_fraction"] == [0.25, 0.0]
    assert report["per_query_boundary_disagreement_fraction"] == [0.25, 0.0]
    assert report["per_query_score_delta"] == [0.25, -0.125]
    assert report["disagreement_fraction"] == 0.125
    assert report["max_absolute_score_delta"] == 0.25
    assert "passed" not in report


@pytest.mark.parametrize("change", ["query_count", "mask_size", "score_count", "nonbinary", "nonfinite"])
def test_comparison_rejects_misaligned_or_invalid_outputs(change):
    baseline = {"masks": np.zeros((2, 2, 2), dtype=np.uint8), "scores": np.zeros(2, dtype=np.float32)}
    candidate = {name: value.copy() for name, value in baseline.items()}
    if change == "query_count":
        candidate = {"masks": np.zeros((1, 2, 2), dtype=np.uint8), "scores": np.zeros(1, dtype=np.float32)}
    elif change == "mask_size":
        candidate["masks"] = np.zeros((2, 3, 2), dtype=np.uint8)
    elif change == "score_count":
        candidate["scores"] = np.zeros(1, dtype=np.float32)
    elif change == "nonbinary":
        candidate["masks"][0, 0, 0] = 2
    else:
        candidate["scores"][0] = np.nan
    with pytest.raises(ValueError):
        benchmark.compare_outputs(baseline, candidate)


def test_dataset_hashes_encoded_bytes_and_freezes_exact_rgb(tmp_path):
    image = tmp_path / "frame.png"
    Image.new("RGBA", (3, 2), (7, 8, 9, 0)).save(image)
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    manifest = tmp_path / "cases.json"
    manifest.write_text(json.dumps({"cases": [{"image": "frame.png", "prompt": "bag of chips", "sha256": digest}]}))
    cases, arrays = benchmark.load_dataset(manifest)
    assert cases[0]["sha256"] == digest
    assert cases[0]["image"] == str(image.resolve())
    assert cases[0]["prompt"] == "bag of chips"
    assert arrays[0].shape == (2, 3, 3)
    assert arrays[0][0, 0].tolist() == [7, 8, 9]
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        benchmark.load_dataset(manifest)


def test_schedule_and_summary_exclude_cold_and_warmup_without_summing_total():
    assert benchmark.sample_schedule(2, repeats=2, warmup=1) == [
        ("cold", 0, 0), ("warmup", 0, 0), ("warmup", 0, 1),
        ("warm", 0, 0), ("warm", 0, 1), ("warm", 1, 0), ("warm", 1, 1),
    ]
    samples = [
        {"phase": "cold", "wall_ms": 9000, "stages_ms": {"pipeline_total": 8000}},
        {"phase": "warmup", "wall_ms": 5000, "stages_ms": {"pipeline_total": 4000}},
        {"phase": "warm", "wall_ms": 10, "stages_ms": {"pipeline_total": 9, "decoder": 4}},
        {"phase": "warm", "wall_ms": 30, "stages_ms": {"pipeline_total": 29, "decoder": 8}},
    ]
    summary = benchmark.warm_summary(samples)
    assert summary["wall_ms"] == {"count": 2, "median": 20.0, "p95": 29.0}
    assert summary["stages_ms"]["pipeline_total"]["median"] == 19.0
    assert summary["stages_ms"]["decoder"]["median"] == 6.0
    assert "sum" not in summary


def test_case_pairing_checks_order_prompt_and_decoded_rgb_identity():
    cases = [{"index": 0, "image": "/a.png", "sha256": "a" * 64, "prompt": "chips", "rgb_sha256": "b" * 64}]
    baseline = {"cases": cases, "samples": [{"phase": "warm", "repeat": 0, "case_index": 0}]}
    benchmark.validate_pairing(baseline, baseline)
    for key, value in (("prompt", "soup"), ("rgb_sha256", "c" * 64), ("sha256", "d" * 64)):
        candidate = {"cases": [dict(cases[0], **{key: value})], "samples": baseline["samples"]}
        with pytest.raises(ValueError, match="case identity"):
            benchmark.validate_pairing(baseline, candidate)
    with pytest.raises(ValueError, match="sample order"):
        benchmark.validate_pairing(baseline, {"cases": cases, "samples": [{"phase": "warm", "repeat": 1, "case_index": 0}]})


def test_execution_devices_rejects_cpu_and_mixed_fallback():
    for devices in ([], ["CPU"], ["GPU.0", "CPU"], ["AUTO:GPU"]):
        with pytest.raises(RuntimeError, match="GPU"):
            benchmark.require_gpu_execution("decoder", devices)
    assert benchmark.require_gpu_execution("decoder", ["GPU.0"]) == ["GPU.0"]


def test_saved_outputs_and_overlay_keep_all_query_indices(tmp_path):
    output = {"masks": np.array([[[1, 0], [0, 0]], [[0, 1], [0, 0]]], dtype=np.uint8),
              "scores": np.array([0.125, 0.875], dtype=np.float32)}
    record = benchmark.save_outputs(tmp_path / "outputs.npz", output)
    assert record["raw_output_bytes"] == 16
    assert record["file_bytes"] == (tmp_path / "outputs.npz").stat().st_size
    with np.load(tmp_path / "outputs.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["masks"], output["masks"])
        np.testing.assert_array_equal(saved["scores"], output["scores"])
    benchmark.save_overlay(tmp_path / "overlay.png", np.zeros((2, 2, 3), dtype=np.uint8), output)
    with Image.open(tmp_path / "overlay.png") as sheet:
        assert sheet.width > 0 and sheet.height > 0


def test_report_pairs_persisted_arrays_and_rejects_artifact_tampering(tmp_path):
    baseline_output = {"masks": np.array([[[1, 0]]], dtype=np.uint8),
                       "scores": np.array([0.5], dtype=np.float32)}
    candidate_output = {"masks": np.array([[[1, 1]]], dtype=np.uint8),
                        "scores": np.array([0.25], dtype=np.float32)}
    left = benchmark.save_outputs(tmp_path / "left.npz", baseline_output)
    right = benchmark.save_outputs(tmp_path / "right.npz", candidate_output)
    identity = {"phase": "warm", "repeat": 0, "case_index": 0}
    baseline = {"cases": [{"index": 0}], "samples": [{**identity, "wall_ms": 20, "output": left}]}
    candidate = {"cases": [{"index": 0}], "samples": [{**identity, "wall_ms": 10, "output": right}]}
    comparison, = benchmark.build_comparisons(baseline, candidate)
    assert comparison["per_query_iou"] == [0.5]
    assert comparison["per_query_score_delta"] == [-0.25]
    assert comparison["wall_delta_ms"] == -10
    (tmp_path / "right.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        benchmark.build_comparisons(baseline, candidate)


def test_worker_timeout_terminates_child_without_models(tmp_path, monkeypatch):
    sleeper = tmp_path / "sleeper.py"
    sleeper.write_text("import time; time.sleep(60)\n")
    monkeypatch.setattr(benchmark, "__file__", str(sleeper))
    import subprocess
    with pytest.raises(subprocess.TimeoutExpired):
        benchmark.run_worker(tmp_path / "job.json", timeout=0.05)


def test_rpc_gpu_declaration_requires_actual_gpu_metadata_without_claiming_kernels():
    description = "SAM3 OpenVINO GPU; all proposals; native binary masks; CPU preprocessing/geometry glue"
    evidence = benchmark.validate_service_metadata({"description": description})
    assert evidence["declared_device"] == "GPU"
    assert evidence["actual_kernel_precision"] == "not_measured"
    assert evidence["per_network_properties"] == "not_exposed_by_RPC"
    for value in ("SAM3 OpenVINO CPU; all proposals", "", "GPU"):
        with pytest.raises(RuntimeError, match="GPU"):
            benchmark.validate_service_metadata({"description": value})


def test_offline_before_after_reports_require_same_settings_and_keep_coverage(tmp_path):
    output = {"masks": np.array([[[1, 0]]], dtype=np.uint8), "scores": np.array([0.5], dtype=np.float32)}
    saved = benchmark.save_outputs(tmp_path / "native.npz", output)
    sample = {"phase": "warm", "repeat": 0, "case_index": 0, "wall_ms": 10,
              "stages_ms": {}, "output": saved}
    shared = {"status": "ok", "dataset_sha256": "d" * 64, "dataset_coverage": {"warning": "near duplicates; pilot only"},
              "settings": {"mode": "rpc", "address": "robot.example", "service": "sam3-openvino",
                           "device": "GPU", "repeats": 1, "warmup": 1, "infer_timeout_seconds": 300},
              "limitations": ["Not kernel timing"]}
    variant = {"status": "ok", "cases": [{"index": 0, "sha256": "a" * 64, "frozen_rgb": "/before.npy"}],
               "samples": [sample], "host": "client", "account_uid": 501, "versions": {"python": "3"},
               "warm": {"wall_ms": {"median": 10}}}
    baseline = {**shared, "cases": variant["cases"], "variants": {"baseline": variant}}
    candidate_variant = {**variant, "cases": [{**variant["cases"][0], "frozen_rgb": "/after.npy"}]}
    candidate = {**shared, "variants": {"candidate": candidate_variant}}
    paired = benchmark.compare_reports(baseline, candidate)
    assert paired["dataset_coverage"] == {"warning": "near duplicates; pilot only"}
    assert paired["performance"]["warm_median_ratio_candidate_over_baseline"] == 1.0
    assert paired["quality_verdict"] == "requires human review"
    for key, value in (("service", "different"), ("device", "CPU"), ("warmup", 0), ("repeats", 2)):
        changed = {**candidate, "settings": {**shared["settings"], key: value}}
        with pytest.raises(ValueError, match="settings"):
            benchmark.compare_reports(baseline, changed)
    with pytest.raises(ValueError, match="dataset"):
        benchmark.compare_reports(baseline, {**candidate, "dataset_sha256": "e" * 64})


def test_errors_do_not_persist_environment_api_credentials(monkeypatch):
    monkeypatch.setenv("VIAM_API_KEY", "secret-key-value")
    monkeypatch.setenv("VIAM_API_KEY_ID", "secret-key-id")
    text = benchmark.safe_error(RuntimeError("secret-key-value / secret-key-id failed"))
    assert text == "RuntimeError: [REDACTED] / [REDACTED] failed"
