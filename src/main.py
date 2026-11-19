import asyncio
import logging
import multiprocessing
import os
import sys

# The OpenVINO wheel bundles openvino_telemetry, which the model-conversion tool imports and which
# reports usage to Google Analytics from a spawned child process. Block it before openvino is
# imported: ovc falls back to its built-in no-op telemetry stub when the package is unavailable.
sys.modules["openvino_telemetry"] = None  # type: ignore[assignment]

from viam.module.module import Module  # noqa: E402

# Importing the model classes registers them with the SDK's resource registry.
from models.detr_vision import RfDetrVision  # noqa: E402, F401
from models.diagnostics import OpenVINODiagnostics  # noqa: E402, F401
from models.openvino_mlmodel import OpenVINOMLModel  # noqa: E402, F401
from models.sam3_mlmodel import Sam3MLModel  # noqa: E402, F401
from models.yolo_vision import YoloVision  # noqa: E402, F401


def _selftest() -> int:
    """Smoke test for packaged binaries: list devices and, if VIAM_OPENVINO_SELFTEST_MODEL is set,
    compile it on CPU and run one inference. Used by CI; not part of the module protocol."""
    from models.config import MLModelConfig
    from models.engine import OpenVINOEngine, list_devices

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("selftest")
    print(list_devices())
    model_path = os.environ.get("VIAM_OPENVINO_SELFTEST_MODEL")
    if model_path:
        cfg = MLModelConfig.from_attributes({"model_path": model_path, "device": "CPU", "cache_dir": ""})
        engine = OpenVINOEngine(cfg, log)
        engine.load()
        print(engine.benchmark(iterations=3, warmup=1))
        engine.close()
    print("selftest ok")
    return 0


def _sam3_import_selftest() -> int:
    import ftfy
    import regex
    import torch
    from torchvision.ops import roi_align

    from models.sam3_reference import SimpleTokenizer  # noqa: F401

    value = roi_align(torch.ones(1, 1, 2, 2), torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0]]), 1)
    if value.item() != 1 or ftfy.fix_text("cafÃ©") != "café" or not regex.fullmatch(r"\p{L}+", "café"):
        raise RuntimeError("SAM3 runtime dependency smoke failed")
    print("SAM3 runtime dependencies ok; no SAM3 model inference was exercised")
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    if "--selftest-sam3-imports" in sys.argv:
        sys.exit(_sam3_import_selftest())
    asyncio.run(Module.run_from_registry())
