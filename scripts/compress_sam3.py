"""Apply the pinned SAM3 notebook's weight-only recipe to a separate FP32 bundle.

Install requirements-sam3-compress.txt in an isolated environment. Encoders use
INT4_SYM/group128/ratio1 with IGNORE fallback for undersized groups; three heads
use INT8_ASYM. Notebook all_layers=False retains first/last eligible layers in
INT8_ASYM backup; ratio1 does not force every weight to INT4. Undersized groups
remain uncompressed by NNCF. Remaining float constants in these five IRs serialize
to FP16, matching ov.save_model defaults. This is not activation quantization.
"""
from __future__ import annotations

import argparse
import copy
import ctypes
import os
from pathlib import Path
import shutil
import sys
import tempfile
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.sam3_bundle import NOTEBOOK_COMPRESSION_MODELS, file_sha256, load_bundle


def _snapshot(root: Path) -> dict[str, str]:
    files = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"bundle must not contain symlinks: {path}")
        if path.is_file():
            files[str(path.relative_to(root))] = file_sha256(path)
        elif not path.is_dir():
            raise ValueError(f"bundle contains a nonregular artifact: {path}")
    return files


def _publish(stage: Path, output: Path) -> None:
    # Directory rename must not replace a concurrently created empty destination.
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "linux":
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        arguments = (-100, os.fsencode(stage), -100, os.fsencode(output), 1)
    elif sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        arguments = (os.fsencode(stage), os.fsencode(output), 4)
    else:
        raise RuntimeError("atomic no-replace bundle publication requires Linux or macOS")
    rename.restype = ctypes.c_int
    if rename(*arguments) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(output))


def compress(model_dir: Path, output: Path) -> None:
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
        raise ValueError("compression requires an uncompressed FP32 source bundle")
    import nncf
    import openvino as ov

    if nncf.__version__ != "3.4.0" or ov.__version__.split("-")[0] != "2026.4.0":
        raise ValueError("use the isolated pinned nncf==3.4.0 and openvino==2026.4.0 compression environment")
    changed = {"manifest.json"}
    for name in NOTEBOOK_COMPRESSION_MODELS:
        relative = Path(bundle.manifest["models"][name])
        changed.update((str(relative), str(relative.with_suffix(".bin"))))
    if len(changed) != 11 or "manifest.json" in bundle.manifest["files"]:
        raise ValueError("compression requires distinct network pairs and a non-self-referencing manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sam3-compress-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        shutil.copytree(source, stage)
        if _snapshot(stage) != original:
            raise ValueError("source bundle changed while copying")
        core = ov.Core()
        for name, settings in NOTEBOOK_COMPRESSION_MODELS.items():
            relative = bundle.manifest["models"][name]
            print(f"Compressing {name}: {settings}", flush=True)
            # Reading the destination would mmap weights that save_model truncates.
            model = core.read_model(source / relative)
            arguments = {"mode": nncf.CompressWeightsMode[settings["mode"]]}
            if settings["mode"] == "INT4_SYM":
                arguments.update(group_size=128, ratio=1.0, advanced_parameters=nncf.AdvancedCompressionParameters(
                    group_size_fallback_mode=nncf.GroupSizeFallbackMode.IGNORE))
            compressed = nncf.compress_weights(model, **arguments)
            ov.save_model(compressed, stage / relative)
            del compressed, model
        manifest = copy.deepcopy(bundle.manifest)
        manifest["precision"] = "MIXED"
        manifest["compression"] = {
            "method": "nncf.compress_weights", "recipe": "notebook_sam3_weight_only",
            "nncf_version": nncf.__version__, "source_manifest_sha256": original["manifest.json"],
            "models": copy.deepcopy(NOTEBOOK_COMPRESSION_MODELS), "serialization": {"compress_to_fp16": True},
        }
        for relative in manifest["files"]:
            manifest["files"][relative] = file_sha256(stage / relative)
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        candidate = _snapshot(stage)
        if candidate.keys() != original.keys() or any(candidate[p] != original[p] for p in original.keys() - changed):
            raise ValueError("compression changed files outside the five-network recipe")
        load_bundle(str(stage))
        if _snapshot(source) != original:
            raise ValueError("source bundle changed during compression; refusing publication")
        _publish(stage, output)
    print(json.dumps({"bundle": str(output), "manifest_sha256": file_sha256(output / "manifest.json"),
                      "source_manifest_sha256": original["manifest.json"],
                      "source_bytes": sum((source / p).stat().st_size for p in original),
                      "candidate_bytes": sum((output / p).stat().st_size for p in original)}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compress(args.model_dir, args.output)


if __name__ == "__main__":
    main()
