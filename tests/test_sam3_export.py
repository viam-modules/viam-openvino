import sys
from types import SimpleNamespace

import pytest

from scripts import export_sam3


@pytest.mark.parametrize("supplied", [False, True])
def test_modified_checkpoint_is_rejected_before_conversion(tmp_path, monkeypatch, supplied):
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.write_bytes(b"modified checkpoint")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        hf_hub_download=lambda *args, **kwargs: str(checkpoint),
        get_hf_file_metadata=lambda *args, **kwargs: SimpleNamespace(
            etag="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
        hf_hub_url=lambda *args, **kwargs: "https://huggingface.co/official-checkpoint",
    ))

    def unexpected_conversion(*args, **kwargs):
        raise AssertionError("modified checkpoint reached conversion")

    monkeypatch.setattr(export_sam3, "export", unexpected_conversion)
    argv = ["export_sam3", "--output", str(tmp_path / "output")]
    if supplied:
        argv += ["--checkpoint", str(checkpoint)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="checkpoint"):
        export_sam3.main()
