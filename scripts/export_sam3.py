"""Export authorized official SAM3 weights into a versioned OpenVINO bundle."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.sam3_bundle import HF_REVISION, MODEL_NAMES, NOTEBOOK_REVISION, SAM3_REVISION, file_sha256, load_bundle


def download(url: str, target: Path) -> None:
    with urllib.request.urlopen(url, timeout=120) as response, target.open("wb") as stream:
        shutil.copyfileobj(response, stream)


def prepare_source(work: Path) -> tuple[Path, Path]:
    archive = work / "sam3.zip"
    download(f"https://github.com/facebookresearch/sam3/archive/{SAM3_REVISION}.zip", archive)
    source = work / "source"
    source.mkdir()
    prefix = f"sam3-{SAM3_REVISION}/"
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            if not member.filename.startswith(prefix) or member.is_dir():
                continue
            relative = Path(member.filename[len(prefix):])
            if not (str(relative).startswith("sam3/") or str(relative) == "LICENSE"):
                continue
            destination = (source / relative).resolve()
            if source.resolve() not in destination.parents:
                raise ValueError("invalid official source archive path")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(member) as stream, destination.open("wb") as target:
                shutil.copyfileobj(stream, target)
    patches = {
        "position_encoding.py": [('device="cuda")', 'device="cpu")')],
        "decoder.py": [('device_type="cuda", enabled=False', 'device_type="cpu", enabled=False'),
                       ('feat_size, feat_size, device="cuda"', 'feat_size, feat_size, device="cpu"')],
        "geometry_encoders.py": [(".pin_memory().to(", ".to(")],
        "sam3_tracker_base.py": [(".pin_memory().to(", ".to(")],
        "sam3_video_inference.py": [(".pin_memory().to(", ".to("),
                                    ('@torch.autocast(device_type="cuda", dtype=torch.bfloat16)',
                                     '# CPU export: CUDA autocast disabled')],
        "sam3_tracking_predictor.py": [('torch.autocast(device_type="cuda", dtype=torch.bfloat16)',
                                        'torch.autocast(device_type="cpu", enabled=False)')],
        "vitdet.py": [("x = addmm_act(type(self.act), self.fc1, x)", "x = self.act(self.fc1(x))")],
        "sam3_tracker_utils.py": [
            ("from sam3.model.edt import edt_triton\n", ""),
            ("    fn_mask_dt = edt_triton(padded_fn_masks)",
             "    from sam3.model.edt import edt_triton\n\n    fn_mask_dt = edt_triton(padded_fn_masks)"),
        ],
    }
    applied = []
    for filename, replacements in patches.items():
        path = source / "sam3" / "model" / filename
        text = path.read_text()
        for old, new in replacements:
            count = text.count(old)
            if count:
                text = text.replace(old, new)
                applied.append({"file": filename, "old": old, "new": new, "count": count})
        path.write_text(text)
    (work / "cpu_patches.json").write_text(json.dumps(applied, indent=2))
    helper = work / "ov_sam3_helper.py"
    download(
        f"https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/{NOTEBOOK_REVISION}"
        "/notebooks/sam3/ov_sam3_helper.py", helper,
    )
    return source, helper


def export(checkpoint: Path, output: Path, checkpoint_revision: str = HF_REVISION) -> None:
    if output.exists():
        raise ValueError(f"output already exists; refusing to replace it: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sam3-export-", dir=output.parent) as temporary:
        work = Path(temporary)
        source, helper_path = prepare_source(work)
        sys.path.insert(0, str(source))
        import torch
        import openvino as ov
        from sam3.model_builder import build_sam3_image_model

        spec = importlib.util.spec_from_file_location("sam3_export_reference", helper_path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        torch.manual_seed(0)
        model = build_sam3_image_model(
            bpe_path=str(source / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
            device="cpu", eval_mode=True, checkpoint_path=str(checkpoint),
            load_from_HF=False, enable_inst_interactivity=False, compile=False,
        ).float().eval()
        helper.patch_vit_rope(model.backbone.vision_backbone.trunk)
        stage = work / "bundle"
        stage.mkdir()
        interfaces = {}

        def convert(name, wrapper, inputs):
            wrapper.eval()
            inputs = inputs if isinstance(inputs, tuple) else (inputs,)
            print(f"Converting {name}", flush=True)
            converted = ov.convert_model(wrapper, example_input=inputs, input=[tuple(t.shape) for t in inputs])
            for prefix, items in (("input", converted.inputs), ("output", converted.outputs)):
                for index, port in enumerate(items):
                    port.get_tensor().set_names({f"{prefix}{index}"})
            path = stage / f"{name}.xml"
            ov.save_model(converted, path, compress_to_fp16=False)
            def ports(items):
                return [{"name": p.get_any_name(), "shape": list(p.shape),
                         "dtype": p.element_type.get_type_name()} for p in items]
            interfaces[name] = {"inputs": ports(converted.inputs), "outputs": ports(converted.outputs)}
            return converted

        z = torch.zeros
        convert("text_encoder", helper.Sam3TextEncoderModel(model.backbone.language_backbone), z(1, 32, dtype=torch.long))
        length, dimension, prompt_length = 72 * 72, 256, 43
        feature, position = z(length, 1, dimension), z(length, 1, dimension)
        prompt, mask = z(prompt_length, 1, dimension), z(1, prompt_length, dtype=torch.bool)
        convert("transformer_encoder", helper.Sam3TransformerEncoderModel(model.transformer.encoder, 72, 72),
                (feature, position, prompt, mask))
        decoder = convert("decoder", helper.Sam3TransformerDecoderModel(model.transformer.decoder),
                (feature, position, z(1), torch.tensor([0]), torch.tensor([[72, 72]]), torch.ones(1, 1, 2), prompt, mask))
        queries = model.transformer.decoder.num_queries
        hs_shape = list(decoder.output(0).shape)
        references_shape = list(decoder.output(1).shape)
        hs_shape[1], hs_shape[2] = hs_shape[2], hs_shape[1]
        references_shape[1], references_shape[2] = references_shape[2], references_shape[1]
        convert("scoring", helper.Sam3ScoringModel(model.dot_prod_scoring, model.transformer.decoder.bbox_embed,
                                                   model.transformer.decoder.norm),
                (z(*hs_shape), z(*references_shape), prompt, mask))
        segmentation = convert("segmentation_head", helper.Sam3SegmentationHeadModel(model.segmentation_head),
                (z(1, 256, 288, 288), z(1, 256, 144, 144), z(1, 256, 72, 72),
                 z(1, queries, 256), feature, prompt, mask))
        geometry = model.geometry_encoder
        helper.save_geo_encoder_config(geometry, str(stage))
        convert("geo_projections", helper.Sam3GeoProjectionsForOV(geometry),
                (z(10, 1, 4), z(10, 1, geometry.pos_enc.num_pos_feats * 2 + 2),
                 z(10, 256, geometry.roi_size, geometry.roi_size), torch.ones(10, 1, dtype=torch.long)))
        convert("geo_cross_attn", helper.Sam3GeoCrossAttnForOV(geometry),
                (z(10, 1, 256), torch.ones(1, 10, dtype=torch.bool), feature, position))
        convert("image_encoder", helper.Sam3ImageEncoderModel(model.backbone.vision_backbone, model.backbone.scalp), z(1, 3, 1008, 1008))
        shutil.copyfile(source / "sam3/assets/bpe_simple_vocab_16e6.txt.gz", stage / "bpe_simple_vocab_16e6.txt.gz")
        shutil.copyfile(source / "LICENSE", stage / "SAM_LICENSE")
        shutil.copyfile(work / "cpu_patches.json", stage / "cpu_patches.json")
        raw_shape = list(segmentation.output(0).shape)
        if len(raw_shape) != 4 or raw_shape[0] != 1 or raw_shape[1] != queries:
            raise ValueError(f"unexpected native mask shape: {raw_shape}")
        manifest = {
            "schema_version": 1, "adapter_version": 1, "model_id": "facebook/sam3",
            "checkpoint_revision": checkpoint_revision, "checkpoint_sha256": file_sha256(checkpoint),
            "sam3_revision": SAM3_REVISION, "notebook_revision": NOTEBOOK_REVISION,
            "precision": "FP32", "image_size": [1008, 1008], "context_length": 32,
            "output_shape": raw_shape[1:], "models": {name: name + ".xml" for name in MODEL_NAMES},
            "interfaces": interfaces, "geo_config": "geo_config.json", "tokenizer": "bpe_simple_vocab_16e6.txt.gz",
            "preprocess": {"resize": "bilinear", "antialias": True, "mean": [0.5] * 3, "std": [0.5] * 3},
            "mask_semantics": "native_logits_gt_zero", "score_semantics": "matching_sigmoid_times_presence_sigmoid",
            "versions": {"openvino": ov.__version__, "torch": torch.__version__},
            "files": {p.name: file_sha256(p) for p in stage.iterdir() if p.is_file()},
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        load_bundle(str(stage))
        os.rename(stage, output)
        print(f"Exported validated official SAM3 bundle: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--hf-revision", default=HF_REVISION, choices=[HF_REVISION])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.checkpoint is None:
        from huggingface_hub import hf_hub_download
        checkpoint = Path(hf_hub_download("facebook/sam3", "sam3.pt", revision=args.hf_revision, token=True))
    else:
        checkpoint = args.checkpoint.resolve()
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    metadata = get_hf_file_metadata(hf_hub_url("facebook/sam3", "sam3.pt", revision=args.hf_revision), token=True)
    if file_sha256(checkpoint) != metadata.etag:
        raise ValueError("checkpoint does not match the pinned official Hugging Face checkpoint")
    export(checkpoint, args.output.resolve(), args.hf_revision)


if __name__ == "__main__":
    main()
