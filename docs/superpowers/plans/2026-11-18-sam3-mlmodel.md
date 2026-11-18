# SAM3 OpenVINO MLModel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the working `viam:openvino:sam3` MLModel service with official-weight provisioning, full-image text inference, all native-resolution proposal masks and scores, and safe single-request execution.

**Architecture:** A separate MLModel resource owns a SAM3 pipeline, not a subclass or replacement of the generic model service. Adapt the pinned official OpenVINO example; reuse compatible existing engine/device/cache utilities and resource conventions without generalizing the engine. Export artifacts separately from startup and verify against the official Meta model.

**Tech Stack:** Python, existing viam-sdk[mlmodel]==0.85.2 and openvino==2026.4.0, NumPy, Pillow, compatible pinned CPU PyTorch/torchvision and SAM3 tokenizer dependencies, pytest/pytest-asyncio, existing PyInstaller build.

**Spec:** `docs/superpowers/specs/2026-11-18-sam3-mlmodel-design.md` (read before executing).

## Global Constraints

- Public model `viam:openvino:sam3`, API `rdk:service:mlmodel`; existing models retain their behavior.
- Inputs: `image: uint8 [H,W,3]` RGB and `prompt: uint8 [L]` UTF-8. One nonempty image/prompt per call; reject invalid input and tokenizer overflow instead of truncating.
- Outputs: `masks: uint8 [N,Hm,Wm]` binary at native mask-head resolution, `scores: float32 [N]` in model query order. No confidence threshold, top-K, NMS, sorting, or empty-mask removal.
- Explicit CPU default, GPU only when requested and verified. No NPU claim, automatic device fallback, original-checkpoint inference fallback, tracking, geometry prompts in the public API, image-feature cache, or camera/depth dependencies.
- One active Infer per resource, zero waiting queue; caller cancellation does not free capacity until actual worker completion. Close rejects admission and asynchronously drains before release.
- Official Meta weights from revision-pinned `facebook/sam3`; revision-pinned official SAM3 source and OpenVINO notebook helpers. No community IR or implicit SAM3.1 substitution.
- Separate export command. No download/conversion during startup/Infer. Exported networks run in OpenVINO; documented CPU PyTorch/torchvision glue includes learned geometry LayerNorm.
- CPU runtime target: Ubuntu 24.04 x86-64. Preserve existing generic module Linux/Windows builds; do not advertise SAM3 Windows/GPU execution before exercising it.
- No remote installs, robot configuration writes, module reloads, deployment, publishing, or motion without separate approval. Do not embed credentials, checkpoints, or downloaded artifacts in git.

## Review Focus

1. A tokenized prompt that fits 32 positions only before adding start/end tokens must be rejected; Task 2 tests the full encoded length.
2. A cancelled RPC must not admit another request while its worker still uses model buffers; Task 3 uses controlled worker completion, not sleeps.
3. A corrupt/mixed bundle must fail before serving, not compile unrelated compatible-looking files; Task 1 tests missing, modified and incompatible artifacts.
4. A real remote reconfiguration removes/recreates the resource; Task 4 tests SDK lifecycle, including reconstruction failure, rather than only an in-place method.
5. Frozen execution can miss lazy runtime imports/assets although generic selftest passes; Task 5 enters the actual packaged SAM3 path.

## File Map

New:
- `src/models/sam3_bundle.py`: artifact/config loading and schema validation; no model execution.
- `src/models/sam3_pipeline.py`: text-only OpenVINO orchestration, request preparation, all-proposal postprocessing.
- `src/models/sam3_mlmodel.py`: MLModel lifecycle, metadata, busy admission and async worker ownership.
- `src/models/sam3_reference.py`: narrowly adapted, licensed/pinned reference helpers needed at runtime; no conversion imports or original model construction on import. Keep upstream attribution alongside adapted code.
- `scripts/export_sam3.py`: real official-checkpoint conversion and atomic complete-bundle creation.
- `scripts/check_sam3.py`: explicit real-model source or frozen/RPC verification, including reference comparison and cold-start measurements. Not a replacement for consumer tests.
- `requirements-sam3-export.txt`: isolated conversion dependencies and immutable source revisions compatible with the existing OpenVINO pin.
- `tests/test_sam3_bundle.py`, `tests/test_sam3_pipeline.py`, `tests/test_sam3_resource.py`: consumer-visible boundaries and lifecycle behavior.

Modify only as required: `requirements.txt`, `requirements-dev.txt`, `setup.sh`, `main.spec`, `src/main.py`, `meta.json`, `README.md`, `.gitignore`, `.github/workflows/test.yml`, `.github/workflows/deploy.yml`. Reuse fixtures in `tests/conftest.py`; add shared helpers only where multiple tests actually need them. No unrelated cleanup.

## Task 1: Reproducible official export and loadable bundle

**Files:** `scripts/export_sam3.py`, `requirements-sam3-export.txt`, `src/models/sam3_bundle.py`, `tests/test_sam3_bundle.py`, `.gitignore`, provisioning section of `README.md`.

**Interfaces:**
- `load_bundle(model_dir: str) -> Sam3Bundle`: validates schema version 1, compatibility, required relative paths/checksums, dimensions and typed preprocessing/tokenizer metadata. Bundle object exposes `manifest`, `model_paths`, `geo_config_path`, `tokenizer_path`; paths stay within model_dir.
- Export CLI: `python scripts/export_sam3.py --checkpoint /path/to/sam3.pt --output /path/to/bundle`. Optional authorized download uses `--hf-revision <immutable revision>` and existing HF auth, never token arguments/logging. The exported manifest always records the resolved official checkpoint identity/checksum.

- [ ] Add bundle failure tests reusing temporary-directory/IR helpers: missing `.bin`, changed artifact checksum, unsupported adapter schema, incompatible dimensions, and paths escaping the bundle fail with actionable errors. Do not test literal JSON wording.
- [ ] Run `venv/bin/python -m pytest tests/test_sam3_bundle.py -q`; confirm relevant failures before implementing validation.
- [ ] Resolve compatible immutable SAM3/notebook commits and official HF model revision, record exact pins and licenses in the exporter/provenance metadata. Check dependency compatibility with OpenVINO 2026.4.0 and existing CI Python versions; preserve generic supported environments rather than raising their Python floor incidentally.
- [ ] Implement export from the official Meta checkpoint using pinned conversion helpers and explicit CPU patches. Produce the eight IR pairs specified by the design, geometry JSON and matching vocabulary. Export the initial baseline with FP32 weights (`compress_to_fp16=False`); record runtime precision separately, with explicit f32 CPU inference for baseline comparisons. No quantization or implicit variant selection.
- [ ] Export into a staging directory and publish the complete bundle only after file/interface validation; a failed export must not replace a usable output. Refuse overwriting existing output by default. Do not inherit the notebook's skip-if-file-exists behavior that can mix old/new models.
- [ ] Write schema-1 manifest containing artifact hashes, checkpoint/source revisions, conversion versions/precision, ordered named submodel interfaces, resize/normalization/tokenizer settings (32 positions including special tokens), and output semantics. Verify actual exported shapes; do not infer them from notebook dummy tensors.
- [ ] Run bundle tests; with authorized weights and an approved isolated export environment, execute the CLI and reload the generated eight IR models. Report actual artifacts/sizes; lack of weight access is an explicit prerequisite, not permission to declare a synthetic export complete.
- [ ] Update README with provisioning command, authentication prerequisites and redistribution warning; ignore generated weights/bundles/caches. Commit only source/docs when commits are authorized.

## Task 2: Text-only inference returning every proposal

**Files:** `src/models/sam3_pipeline.py`, `src/models/sam3_reference.py`, `tests/test_sam3_pipeline.py`, required runtime dependencies in `requirements.txt`.

**Interfaces:**
- `Sam3Pipeline(bundle: Sam3Bundle, device: str = "CPU", cache_dir: str = "")` loads the exported networks; `infer(image: np.ndarray, prompt: str) -> dict[str, np.ndarray]`; `close() -> None` releases owned runtime resources.
- `decode_inputs(input_tensors: Mapping[str, np.ndarray]) -> tuple[np.ndarray, str]` validates the two exact public tensors, UTF-8 and nonempty text/image.
- `postprocess_proposals(mask_logits: np.ndarray, matching_logits: np.ndarray, presence_logits: np.ndarray) -> dict[str, np.ndarray]` validates single-image compatible axes, computes final-layer matching sigmoid multiplied by presence sigmoid, preserves query order, and thresholds native mask logits at >0. Derive N/Hm/Wm from outputs; require finite numeric logits and matching query counts.

- [ ] Write deterministic behavioral tests: distinguish three ordered proposals, retain a low-confidence proposal and an all-zero mask, verify combined presence/matching scores against known values, and verify negative/zero/positive mask-logit boundaries. Test inconsistent axes and nonfinite outputs as errors rather than partially returning a result.
- [ ] Add input tests for RGB shape/dtype, empty image/text, invalid UTF-8, non-ASCII prompt, and token capacity boundaries using the actual tokenizer including two special tokens. Do not silently truncate via the reference tokenizer's default call.
- [ ] Run `venv/bin/python -m pytest tests/test_sam3_pipeline.py -q` to observe failures, then implement these interfaces and the pinned reference adaptation. Reuse existing dtype/device/cache utilities where they fit; do not alter generic MLModelConfig to force a multi-network bundle into it.
- [ ] Preserve reference normalization, image resize, FPN/prompt layout, empty-geometry branch and CPU affine LayerNorm. Own request-local state. Bypass reference confidence filtering and original-resolution mask interpolation, not merely its configured threshold.
- [ ] Keep runtime imports narrow; include tokenizer/runtime helpers explicitly rather than importing conversion code or loading the Meta checkpoint. Verify reusable helper dependencies do not initialize a PyTorch SAM3 model.
- [ ] Run focused tests and a real CPU pipeline smoke against the Task 1 bundle on a non-square image and multiple prompts. Compare native logits and scores/query count with official Meta outputs before confidence filtering, and independently with the adapted notebook path. Record measured max/mean errors, choose justified tolerances, and fail material differences rather than tuning a threshold to hide them.
- [ ] Update README with precise score, native-binary-mask and caller-resizing semantics. Commit source/tests/docs when authorized.

## Task 3: MLModel resource and cancellation-safe single admission

**Files:** `src/models/sam3_mlmodel.py`, `tests/test_sam3_resource.py`, `src/main.py`, `meta.json`, service section of `README.md`.

**Interfaces:** `Sam3MLModel(MLModel, EasyResource)`, `MODEL = "viam:openvino:sam3"`; SDK-compatible `new`, `validate_config`, `infer`, `metadata`, async `close`. Attributes `model_dir` required, `device` CPU default/explicit GPU, `cache_dir` optional (default resource-specific subdirectory under `$VIAM_MODULE_DATA/ov_cache`, blank disables). No confidence threshold or concurrency configuration.

- [ ] Reuse ComponentConfig/metadata fixtures and deterministic worker barriers/events. Add behavioral tests showing a second call fails busy without invoking the pipeline; cancellation of the first caller does not free admission; successful/failed worker completion restores admission; close rejects new calls and waits without stopping an asyncio ticker. No arbitrary sleep-based race assertions.
- [ ] Run `venv/bin/python -m pytest tests/test_sam3_resource.py -q` to observe failures.
- [ ] Implement admission on the resource event loop and one resource-local worker. Shield worker ownership from awaiting-task cancellation; use actual concurrent worker completion to schedule release on the owning loop, including exceptions. Never release admission in the RPC coroutine's finally block just because the RPC ended. No waiting submissions, forced OpenVINO cancellation, retry loop, scheduler, or incidental generic-service changes.
- [ ] Close stops admission, asynchronously waits for the owned worker, then releases pipeline/executor; repeat close is safe. Separate resource instances never share busy state. Ensure worker failures are consumed even if the caller has already cancelled.
- [ ] Implement public metadata: image uint8 [-1,-1,3], prompt uint8 [-1], masks uint8 [N,Hm,Wm] (actual bundle dimensions when known), scores float32 [N], and documented native-resolution semantics. Reject unavailable explicit devices; do not fall back.
- [ ] Register the class through the existing entrypoint pattern and add its model metadata without changing current registrations. Keep SAM3-only heavy runtime imports out of unrelated model paths.
- [ ] Run resource tests and source MLModel RPC smoke using Task 2 real pipeline. Assert actual tensor transport/dtypes, low-score retention and native dimensions rather than only successful return. Update README with config, busy behavior and cancellation limits; commit when authorized.

## Task 4: Actual SDK lifecycle, cold startup and transport proof

**Files:** `scripts/check_sam3.py`, lifecycle scenarios in `tests/test_sam3_resource.py`, verification instructions in `README.md`.

**Interfaces:** `python scripts/check_sam3.py --model-dir <bundle> --image <image> --prompt <text> --device CPU --module-entrypoint <path> --report <json>`. Use SDK module RPCs over a local throwaway connection, not robot configuration. Report startup time/cache state, Infer end-to-end latency, proposal/native shape/dtypes, output bytes and peak memory. Exit nonzero for contract/lifecycle/parity failures; output is evidence, not generated unit-test fixtures.

- [ ] Add a deterministic test invoking the pinned SDK remove/recreate path with controlled active work: old work must complete before its pipeline is closed, rejected calls must not execute, failed replacement must surface, and the event loop must remain responsive. Do not reuse direct r.reconfigure() as production lifecycle proof.
- [ ] Run the focused lifecycle test failing before the necessary integration handling, implement only missing behavior, then run it passing.
- [ ] Implement the checker to call AddResource with an empty resource-specific compiled cache and measure total configuration. Invoke the module through its real protocol rather than constructor-only timing. Record configured deadline and fail/report if cold startup exceeds it; document a required explicit timeout adjustment rather than assuming warm-cache performance.
- [ ] Through actual MLModel RPC, exercise the same image with a matching prompt and a no-match prompt, then a different image. Preserve all proposals, inspect resized binary overlays and compare native outputs to the official reference. Exercise busy and caller-cancellation behavior without any arm motion.
- [ ] Measure actual protobuf payloads and peak memory, not only NumPy array sizes. Record cold/warm startup and CPU inference separately. Use approved local/isolated Linux execution; request approval before installing on or deploying to the inspected robot.
- [ ] Record commands/results and limitations in verification notes/README as appropriate; do not claim GPU or measured latency before exercising them. Commit source/docs when authorized.

## Task 5: Frozen packaging and regression coverage

**Files:** `main.spec`, runtime dependency/setup files where necessary, `src/main.py`, `.github/workflows/test.yml`, `.github/workflows/deploy.yml`, `README.md`.

- [ ] Include necessary SAM3 runtime Python modules, tokenizer dependencies/assets and PyTorch/torchvision native libraries in the frozen build. Preserve existing OpenVINO plugin collection and host C++ runtime handling. Keep model bundles external and conversion dependencies out of the deployed executable.
- [ ] Add an explicit SAM3 packaged-smoke mode accepting an external complete bundle and image/prompt; exercise the real pipeline and output contract. Retain existing generic `--selftest`. Do not treat tiny fake networks or import-only checks as real-model SAM3 proof.
- [ ] Extend CI to preserve current generic Linux/Windows builds and smoke. Add gated real-bundle verification only where authorized assets are available; unavailable gated assets must be reported as not exercised, not a passing real-model check. Exercise SAM3 lazy dependency loading in frozen builds even when gated inference is unavailable, but label it as dependency coverage only.
- [ ] Run `venv/bin/python -m pytest -q` once after integration, followed by `./build.sh` in a supported isolated build environment and existing generic frozen smoke. Then run Task 4 checker against the frozen executable with real bundle/image, including cold AddResource and actual Infer.
- [ ] Confirm generic services still load on both existing release platforms; if SAM3 itself is only verified on Linux, document that restriction explicitly rather than breaking Windows packaging or advertising untested support.
- [ ] Update README with export/load/runtime dependency boundaries, artifact compatibility, observed CPU startup/inference results, native-mask limitation and GPU verification status. Remove throwaway scaffolding and keep only useful verification tools. Commit when authorized.

## Task 6: Final acceptance and optional GPU verification

- [ ] Review all design requirements against exercised evidence: official provenance, complete compatible bundle, all proposal/order invariants, original-image state isolation, busy/cancellation/close behavior, actual SDK recreation, cold startup and frozen inference. A missing real-model prerequisite blocks completion; passing isolated tests alone does not complete the feature.
- [ ] Obtain an independent whole-change review and resolve concrete findings. Do not rerun tests merely to restate already reported failures; fix the root cause first.
- [ ] After separate approval for target-machine installs/deployment, verify OpenVINO-visible devices/drivers and execute the same real smoke with `device=GPU`. Compare output parity and measured latency/memory. Explicit GPU failure must stay a failure, not silently run CPU. CPU completion can be reported separately without claiming GPU acceptance.
- [ ] Deliver changed paths, exact exercised commands and results, export provenance, platform/device coverage and outstanding external prerequisites. No robot motion or registry publishing is required.

## Self-review

Coverage: Tasks 1–2 implement official export, provenance, tokenizer and native all-proposal inference; Task 3 implements public API/admission/cleanup; Task 4 proves actual SDK startup/lifecycle/transport; Task 5 proves packaging and existing services; Task 6 closes review and explicitly separates later GPU approval. All five Review Focus conditions have named tests or actual executable smoke in their owning task. No resource changes, installs, model downloads or implementation have been performed by writing this plan.

## Execution outcome

CPU implementation and acceptance are complete. The checkboxes above preserve the original execution checklist; this section records the exercised outcome and explicit limits.

| Task | Outcome |
|---|---|
| 1 — official export/bundle | Complete. Authorized official checkpoint exported on native macOS CPU; all eight IRs, source pins, checksums, metadata and actual interfaces validated. |
| 2 — all-proposal inference | Complete with user-approved numerical exception below. Native masks/scores, full-vocabulary tokenization, preprocessing and proposal alignment compared with Meta and the notebook. |
| 3 — resource/admission | Complete. Controlled tests and real RPC prove busy rejection, cancellation ownership and cleanup. |
| 4 — SDK lifecycle/transport | Complete. Real source and target frozen RPC prove cold startup, actual remove/recreate and failure, no-match and second-image state isolation. |
| 5 — packaging/regressions | Complete for CPU/Linux. Native target frozen executable built and exercised with real weights; all inherited notices verified inside the executable. Linux suite: 155 passed, one existing opt-in YOLO model test skipped. Windows CI paths updated but not executed here; no Windows SAM3 inference claim. |
| 6 — final review | Independent whole-change review found no functional defects. Its two evidence gates were resolved by real frozen execution and the user's explicit numerical acceptance. GPU remains unexercised as planned. |

### Numerical acceptance

Adapter and notebook raw mask/matching/presence logits match exactly. Against original Meta SAM3 on the pinned real image/prompt, preprocessing/token IDs match, confidence checks pass (maximum absolute difference approximately 2.49e-5), and three of 16,588,800 binary mask pixels differ. The stricter raw-logit diagnostic fails for 56 values (maximum absolute difference 0.025154, mean 0.000146). The user explicitly selected acceptance with these measurements documented. No diagnostic threshold was relaxed; the raw report remains failed and the comparison command retains its nonzero exit for that case.

### Exercised commands and evidence

- Official export: native export environment `python scripts/export_sam3.py --checkpoint <authorized pinned sam3.pt> --output <bundle>`; complete validated bundle produced.
- Regression suite: isolated target runtime `venv/bin/python -m pytest -q` -> 155 passed, 1 skipped. The skipped check is `test_yolov8n_end_to_end`, requiring the optional `YOLOV8N_ONNX` artifact.
- Target build: isolated target runtime `venv/bin/python -m PyInstaller --clean --noconfirm main.spec` -> successful Linux executable.
- Target smoke: `scripts/check_sam3.py --model-dir sam3-artifacts --image <pinned official test image> --second-image <public truck image> --prompt shoe --device CPU --module-entrypoint dist/main --report frozen-rpc.json --overlay --exercise-lifecycle` -> passed. Cold AddResource 8.45s; first Infer round trip 21.15s, observed during validation rather than a throughput benchmark.
- Independent comparison: separate target export environment `scripts/compare_sam3.py --model-dir sam3-artifacts --checkpoint sam3.pt --prompt shoe --report reference-comparison.json` -> preserved strict diagnostic failure; public mask/score agreement accepted explicitly as above.
- Target executable `--selftest` and `--selftest-sam3-imports` -> passed; dependency smoke is not substituted for the real-weight RPC run.

Local reports are retained under `.superpowers/sdd/2026-11-18-sam3-mlmodel/`: `intel-frozen-rpc.json`, `intel-reference-comparison.json`, and overlays. The approved isolated target workspace is `/home/viam/sam3-mlmodel-verify.4IcJVc`; its `sam3-artifacts/` bundle and `dist/main` executable are preserved. No running robot configuration, module reload, motion, registry publication or deployment was performed. No commits or push were authorized; source remains on `feat/sam3-mlmodel`.

