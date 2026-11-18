# SAM3 OpenVINO MLModel service

Status: approved by the user, including the adversarial-review revisions and single-active-request/no-queue policy. Implementation planning follows this design.

## Intent and agreed scope

Add `viam:openvino:sam3`, implementing `rdk:service:mlmodel`, to viam-modules/viam-openvino. Keep sam3-detector and existing OpenVINO services unchanged. Target Ubuntu 24.04 x86-64 on the inspected Intel Core Ultra machine. Start with explicit CPU execution; GPU requires separate device/driver and model execution verification. No NPU support claim.

One RGB image and one raw text prompt per Infer call. Perform full inference each time; no feature cache, tracking, camera dependency, depth projection, or grasp planning. Return all model proposals without confidence filtering, top-K, NMS, sorting, or empty-mask removal. Masks use native mask-head spatial resolution, not original image resolution.

## Approaches

1. New standalone repository: isolates dependencies but duplicates existing OpenVINO packaging and lifecycle infrastructure.
2. Extend generic viam:openvino:mlmodel: couples a single-network inference service to SAM3 tokenization and multi-network orchestration.
3. Selected: new SAM3-specific MLModel resource inside viam-openvino. Share established utilities where their contracts fit; preserve the generic resource and other models.

## Public contract

Inputs are named numeric tensors:
- `image`: uint8 [H,W,3], RGB, one nonempty image, no batch dimension.
- `prompt`: uint8 [L], nonempty UTF-8 text, decoded and tokenized internally using the SAM3 tokenizer and its supported context length. Report invalid UTF-8, unsupported shapes/dtypes, and prompts exceeding tokenizer capacity as errors rather than silently changing their meaning.

Outputs:
- `masks`: uint8 [N,Hm,Wm], values 0 or 1, native mask-head resolution.
- `scores`: float32 [N], final-layer per-proposal confidence, aligned with masks in original model query order.

N is obtained from model outputs, not a hardcoded count. Low-confidence and empty masks remain. An image with no convincing matches still returns proposals; it does not become N=0 through filtering. Reject malformed or inconsistent model outputs. Metadata declares fixed channels/dtypes and dynamic dimensions, including the native-resolution mask contract.

Derive confidence from the model's final matching logits and presence prediction using SAM3 semantics. Apply sigmoid to native mask logits and threshold at >0.5 (equivalently logits >0). No original-resolution interpolation happens in the service. To map masks to an input image, callers retain H/W and resize using the documented interpolation and threshold order. For parity with the reference's original-resolution binary masks, callers need mask probabilities/logits, which this binary-only contract does not expose; resizing binary masks is an approximation. Document that distinction explicitly.

## Runtime and reuse

Implement a new SAM3 resource plus a SAM3 pipeline; do not subclass the generic single-model service merely to override its internals. Follow existing lifecycle, validation, executor, metadata, and cleanup conventions. Reuse engine/device/cache utilities where compatible without forcing SAM3 into unrelated generic configuration or adding a generalized pipeline framework.

Use a revision-pinned, attributed adaptation of the official `openvinotoolkit/openvino_notebooks` SAM3 text-prompt pipeline and conversion helpers rather than reimplementing SAM3 mathematics. This is an official OpenVINO-project example, explicitly marked experimental, not a production-validated SAM3 release. Preserve applicable upstream licenses. The inspected helper still uses geometry-encoder processing for empty box prompts; text-only does not justify dropping those stages. Exclude SAM1 interactive and tracking-only artifacts.

The exported neural networks execute through OpenVINO. Preprocessing and geometry glue may execute on the CPU using PyTorch/torchvision, including the geometry branch's learned affine LayerNorm with parameters loaded from `geo_config.json`. Selecting GPU changes the compiled networks' device; it does not promise end-to-end GPU execution. Preserve this reference behavior rather than adding another exported network solely to eliminate that CPU operation. Keep conversion and original checkpoint loading out of resource startup and inference. Avoid importing SAM3-specific dependencies on unrelated service paths. Packaging must explicitly include required runtime dependencies/assets; lazy imports alone do not eliminate package-size costs. Verify frozen packaging, not only source execution.

Configuration: `model_dir` points to a complete, versioned artifact bundle; `device` defaults to CPU and initially accepts CPU or GPU explicitly. Reuse compiled-model caching where compatible. No confidence threshold setting and no silent device or PyTorch-model inference fallback. Validate files, bundle version, input/output ports and device availability with actionable errors. Export records source revisions, preprocessing, tokenizer parameters, native output conventions, and model identities so incompatible submodels cannot be mixed silently.

Use the reference image normalization/resizing and tokenizer exactly. Keep request-local image/prompt state. Allow one active full-pipeline inference per resource and no waiting queue: reject concurrent requests immediately with a clear busy error. Reject requests when closing. Offload blocking work from the event loop using the existing resource pattern.

Tie the busy guard and resource ownership to actual worker completion, not the lifetime of the coroutine awaiting it. If a caller times out or disconnects after work starts, let the computation finish without attempting to interrupt OpenVINO; retain the busy guard until the worker exits, including error paths. Do not enqueue retries or implement a scheduler. Removal/close stops admission, waits for active worker completion asynchronously, then releases compiled resources and the executor without blocking the module event loop. Never release a pipeline still in use or allow an old worker to clear a newly created resource's guard.

## Provisioning

### Authoritative sources

- Weights: official Meta `facebook/sam3` on Hugging Face, pinned to an immutable revision, with authorized access and the applicable SAM license. Use the checkpoint format expected by the selected exporter (`sam3.pt` for the notebook's Meta-source path); do not download both checkpoint formats unnecessarily.
- Model implementation and tokenizer code: official `facebookresearch/sam3`, pinned to a compatible commit. Use that tokenizer's matching vocabulary asset; do not substitute a different Hugging Face tokenizer representation without parity verification.
- Export and runtime adaptation reference: official `openvinotoolkit/openvino_notebooks`, pinned to a compatible commit, including its SAM3 helper. A Jupyter installation or running notebook is not required.
- The inspected official Hugging Face repository contains original checkpoints/configs/tokenizer files, not ready-to-use OpenVINO IR. Downloading the official checkpoint does not eliminate conversion.
- Do not adopt third-party pre-exported Hugging Face IR by default. In particular, the inspected `rajeshgangireddy/SAM3_OpenVINO` card identifies SAM3.1 as its base; changing to it would change both provenance and potentially model version. Require a separate design decision before using community exports or switching SAM versions.

### Reproducible export

Provide a separately invoked export/provisioning command that downloads or consumes the authorized official checkpoint and generates the bundle using the pinned conversion code. Record necessary CPU/conversion patches explicitly; do not fetch or mutate moving upstream source at runtime. Run export in an isolated environment with compatible pinned conversion dependencies. Reproducible means traceable inputs and a repeatable procedure; byte-identical exports are not assumed.

The network split and auxiliary configuration are exporter-specific interfaces, not a standard SAM3 distribution format. Keep exporter and inference adapter versioned together. Never mix files from different exporters, model versions, or precision variants merely because filenames match.

The selected split export produces eight `.xml`/`.bin` pairs: `image_encoder`, `text_encoder`, `geo_projections`, `geo_cross_attn`, `transformer_encoder`, `decoder`, `scoring`, and `segmentation_head`. Include `geo_config.json`, the matching `bpe_simple_vocab_16e6.txt.gz`, and a module-defined `manifest.json`. The current helper stores geometry normalization arrays in `geo_config.json`; a separate `geo_constants.npz` is not required by its actual save/load implementation.

The manifest records bundle schema/adapter version, model identity, official checkpoint revision and checksum, SAM3 and notebook commits, dependency/export versions, precision, relative artifact paths/checksums, submodel ports/shapes/dtypes, image preprocessing, tokenizer/context settings, and native mask/score semantics. Validate the bundle before serving. Select one explicit baseline precision and record it; no implicit quantized-variant selection.

Artifact generation is not performed during resource startup or Infer. Startup loads the bundle and compiles it for the explicitly selected device, optionally using a separate local compiled cache. Neither the original checkpoint, export toolchain, access credentials, nor device-specific cache belongs in the deployed model bundle. Provisioning must not redistribute gated or derivative model assets without the necessary rights. Missing weights or authorization is an explicit prerequisite, not a trigger for a community download, dummy inference, or a PyTorch inference fallback.

The remote machine has no OpenVINO in its system Python; isolated validation dependencies will be needed. Remote installations, deployment, module reloads, configuration writes, and motion require separate approval; no robot motion is needed for this work.

## Verification and acceptance

- Contract tests exercise UTF-8 decoding, shape/dtype errors, all-proposal preservation (including low scores and empty masks), score/mask alignment, mask threshold boundary, and request isolation using existing test conventions. Verify immediate busy rejection, caller cancellation while the worker continues, no admission until actual worker completion, guard release after worker success/failure, and rejection during close.
- Exercise the pinned SDK's actual remove/recreate reconfiguration lifecycle, not just direct calls to the resource's in-place reconfigure method. Verify that active work finishes before old resources are released, new calls are rejected during close, and failed recreation is reported correctly. There are no queued requests to migrate. Cleanup must not block the module event loop or use resources after release.
- Real CPU smoke: export/load actual SAM3 artifacts, call Infer via the MLModel service with a real image/prompt, and inspect native masks, confidences, metadata and proposal count against the reference before filtering. Record latency and peak memory; make no unmeasured performance promise.
- Cold-start acceptance must invoke AddResource through the actual module with an empty compiled-model cache. Measure bundle validation, compilation, total resource-configuration time and peak memory separately from inference. Check against the configured Viam resource-configuration deadline; document an explicit timeout adjustment if required. Standalone construction or warm-cache success is not sufficient startup proof.
- Verify preprocessing, tokenization, confidence/presence semantics and native logits against reference outputs within documented numerical tolerance; inspect a visual overlay made by resizing returned binary masks.
- Establish correctness against the official Meta SAM3 implementation using the same checkpoint, image preprocessing and tokenization, inspecting query-aligned outputs before confidence filtering. Compare the notebook-derived OpenVINO path and the MLModel adapter separately so sharing notebook code does not become the only correctness oracle. Report unsupported operations, conversion failures, or material parity differences rather than treating official-example provenance as proof.
- Verify artifact provenance and compatibility: pinned official downloads, manifest/checksum and interface validation, actionable failures for incomplete or mismatched bundles, and no model downloads/conversion during startup or inference. Use the shared test patterns for consumer-visible failure cases rather than tests of manifest wording.
- Exercise a no-match prompt without suppressing proposals, then a second prompt/image to detect state leakage.
- Run affected existing tests and a packaged entrypoint smoke to ensure other registered services still load and SAM3 assets/dependencies are available.
- GPU is a later explicit verification on the target machine; CPU success is not GPU proof.

## Risks and builder decisions

The upstream notebook is an official OpenVINO example but explicitly experimental. Official Meta weights establish model provenance, not OpenVINO conversion correctness. Correct conversion and submodel interfaces must be proven against official-model inference; there is no promise that a downloaded third-party IR is interchangeable. Native binary masks trade smaller payloads for loss of subpixel probability information, especially at boundaries. CPU latency and memory are unknown until measured. Weight access, approved installation environment, and reference image availability may gate real-model verification.

Builder chooses internal filenames, exact pinned revisions after compatibility checks, safe executor ownership mechanics, and numerical tolerances based on measured reference behavior. Do not expand the public outputs, add performance features, or alter other services as incidental cleanup.

## Pre-implementation review

Initial builder and integration reviews found no material planning blockers. A subsequent adversarial review identified the learned CPU normalization exception and lifecycle/startup acceptance gaps. The user accepted the documented CPU glue, actual SDK remove/recreate verification, cold AddResource verification, and a single-active-request/no-queue policy. The user approved the resulting design. These reviews and approval do not establish real-model correctness, performance, or packaged execution.
