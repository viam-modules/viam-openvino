# Copyright (c) OpenVINO contributors. All rights reserved.
# Orchestration/geometry adapted from openvino_notebooks SAM3 helper at 343266b9537c5bfb2aad86916fbe0f2c2569250f (Apache-2.0; see LICENSE).
"""Text-only SAM3 input validation and native-resolution proposal outputs."""

from __future__ import annotations

from typing import Mapping

import numpy as np


def decode_inputs(input_tensors: Mapping[str, np.ndarray]) -> tuple[np.ndarray, str]:
    if set(input_tensors) != {"image", "prompt"}:
        raise ValueError("SAM3 requires exactly the image and prompt tensors")
    image, encoded = input_tensors["image"], input_tensors["prompt"]
    if (
        not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
        or image.shape[2] != 3 or min(image.shape[:2]) <= 0
    ):
        raise ValueError("image must be a nonempty RGB uint8 [H,W,3] tensor")
    if not isinstance(encoded, np.ndarray) or encoded.dtype != np.uint8 or encoded.ndim != 1 or encoded.size == 0:
        raise ValueError("prompt must be a nonempty UTF-8 uint8 [L] tensor")
    try:
        prompt = encoded.tobytes().decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("prompt must contain valid UTF-8 text") from exc
    if not prompt.strip():
        raise ValueError("prompt must contain non-whitespace text")
    return image, prompt


def postprocess_proposals(
    mask_logits: np.ndarray, matching_logits: np.ndarray, presence_logits: np.ndarray,
) -> dict[str, np.ndarray]:
    for value in (mask_logits, matching_logits, presence_logits):
        if not isinstance(value, np.ndarray) or value.dtype.kind != "f" or not np.isfinite(value).all():
            raise ValueError("SAM3 logits must be finite floating-point arrays")
    if mask_logits.ndim != 4 or mask_logits.shape[0] != 1 or min(mask_logits.shape[1:]) <= 0:
        raise ValueError("mask logits must have shape [1,N,Hm,Wm]")
    count = mask_logits.shape[1]
    if (
        matching_logits.ndim != 4 or matching_logits.shape[0] <= 0
        or matching_logits.shape[1:] != (1, count, 1)
        or presence_logits.shape != (matching_logits.shape[0], 1, 1)
    ):
        raise ValueError("matching/presence logits must share decoder layers and proposal count")
    matching = matching_logits[-1, 0, :, 0].astype(np.float32, copy=False)
    presence = np.float32(presence_logits[-1, 0, 0])
    scores = np.exp(-np.logaddexp(np.float32(0), -matching)) * np.exp(-np.logaddexp(np.float32(0), -presence))
    return {"masks": (mask_logits[0] > 0).astype(np.uint8), "scores": scores.astype(np.float32, copy=False)}


def tokenize_prompt(tokenizer, prompt: str) -> np.ndarray:
    ids = [tokenizer.sot_token_id, *tokenizer.encode(prompt), tokenizer.eot_token_id]
    if len(ids) > 32:
        raise ValueError("prompt exceeds the 32-token context including start/end tokens")
    tokens = np.zeros((1, 32), dtype=np.int64)
    tokens[0, :len(ids)] = ids
    return tokens


class Sam3Pipeline:
    def __init__(self, bundle, device: str = "CPU", cache_dir: str = ""):
        import json
        from pathlib import Path

        import openvino as ov
        import torch
        from torchvision.transforms import v2

        from .engine import _device_available
        from .sam3_reference import SimpleTokenizer

        if device not in ("CPU", "GPU"):
            raise ValueError("SAM3 device must be CPU or GPU")
        self._core = ov.Core()
        if not _device_available(device, self._core.available_devices):
            raise RuntimeError(f"requested device {device} is unavailable: {self._core.available_devices}")
        self._compiled = {}
        self._requests = {}
        self._closed = False
        self._bundle = bundle
        config = {"PERFORMANCE_HINT": "LATENCY"}
        if device == "CPU":
            config["INFERENCE_PRECISION_HINT"] = "f32"
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            config["CACHE_DIR"] = cache_dir
        try:
            for name, path in bundle.model_paths.items():
                compiled = self._core.compile_model(str(path), device, config)
                self._compiled[name] = compiled
                self._requests[name] = compiled.create_infer_request()
            self._tokenizer = SimpleTokenizer(bundle.tokenizer_path)
            self._geo = json.loads(bundle.geo_config_path.read_text())
            self._norm_weight = torch.tensor(self._geo["img_pre_norm_weight"], dtype=torch.float32)
            self._norm_bias = torch.tensor(self._geo["img_pre_norm_bias"], dtype=torch.float32)
            self._transform = v2.Compose([
                v2.ToImage(), v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(tuple(bundle.manifest["image_size"]), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ])
        except BaseException:
            self.close()
            raise

    def _run(self, name, inputs):
        request = self._requests[name]
        request.infer(inputs)
        # Each network runs once per serialized request; views remain valid until the next request.
        return [np.asarray(request.get_output_tensor(index).data) for index in range(len(self._compiled[name].outputs))]

    def _geometry(self, feature, position, height, width):
        import torch
        import torch.nn.functional as functional
        from torchvision.ops import roi_align

        cfg = self._geo
        dimensions = cfg["d_model"]
        boxes = torch.zeros(10, 1, 4)
        cx, cy, box_width, box_height = (boxes[:, :, index].flatten() for index in range(4))
        periods = torch.arange(cfg["pos_enc_num_feats"], dtype=torch.float32)
        periods = cfg["pos_enc_temperature"] ** (2 * (periods // 2) / cfg["pos_enc_num_feats"])
        x = cx[:, None] * cfg["pos_enc_scale"] / periods
        y = cy[:, None] * cfg["pos_enc_scale"] / periods
        x = torch.stack((x[:, 0::2].sin(), x[:, 1::2].cos()), dim=2).flatten(1)
        y = torch.stack((y[:, 0::2].sin(), y[:, 1::2].cos()), dim=2).flatten(1)
        positions = torch.cat((y, x, box_height[:, None], box_width[:, None]), dim=1).view(10, 1, -1)
        normalized = functional.layer_norm(torch.from_numpy(feature), [dimensions], self._norm_weight, self._norm_bias)
        normalized = normalized.permute(1, 2, 0).reshape(1, dimensions, height, width)
        roi = roi_align(normalized, torch.zeros(10, 5), cfg["roi_size"])
        projected = self._run("geo_projections", [
            boxes.numpy(), positions.numpy(), roi.numpy(), np.ones((10, 1), dtype=np.int64),
        ])[0]
        return self._run("geo_cross_attn", [projected, np.ones((1, 10), dtype=bool), feature, position])

    def _forward(self, image: np.ndarray, prompt: str):
        import torch

        if self._closed:
            raise RuntimeError("SAM3 pipeline is closed")
        with torch.inference_mode():
            tokens = tokenize_prompt(self._tokenizer, prompt)
            image_tensor = self._transform(image).unsqueeze(0).numpy()
            image_outputs = self._run("image_encoder", [image_tensor])
            fpn, positions = image_outputs[:3], image_outputs[3:6]
            height, width = fpn[2].shape[-2:]
            feature = fpn[2].reshape(1, 256, -1).transpose(2, 0, 1)
            position = positions[2].reshape(1, 256, -1).transpose(2, 0, 1)
            text, text_mask = self._run("text_encoder", [tokens])
            geometry, geometry_mask = self._geometry(feature, position, height, width)
            prompt_features = np.concatenate((text, geometry), axis=0)
            prompt_mask = np.concatenate((text_mask, geometry_mask), axis=1).astype(bool, copy=False)
            encoded = self._run("transformer_encoder", [feature, position, prompt_features, prompt_mask])
            decoded = self._run("decoder", [
                encoded[0], encoded[1], np.zeros(1, dtype=np.float32),
                encoded[3], encoded[4], encoded[5], prompt_features, prompt_mask,
            ])
            queries, references = decoded[0].transpose(0, 2, 1, 3), decoded[1].transpose(0, 2, 1, 3)
            matching = self._run("scoring", [queries, references, prompt_features, prompt_mask])[0]
            masks = self._run("segmentation_head", [
                *fpn, queries[-1], encoded[0], prompt_features, prompt_mask,
            ])[0]
            return masks, matching, decoded[2]

    def infer(self, image: np.ndarray, prompt: str) -> dict[str, np.ndarray]:
        result = postprocess_proposals(*self._forward(image, prompt))
        if tuple(result["masks"].shape) != tuple(self._bundle.manifest["output_shape"]):
            raise ValueError("native proposal masks differ from the validated bundle output_shape")
        return result

    def close(self) -> None:
        self._closed = True
        self._requests.clear()
        self._compiled.clear()
        self._core = None
