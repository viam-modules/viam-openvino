import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="Linux provisioning uses a POSIX shell")


def run_gpu_setup(tmp_path, *, installed=True, accessible=True, enumerated=False, mode="auto"):
    root = tmp_path / "host"
    pci = root / "sys/bus/pci/devices/0000:00:02.0"
    pci.mkdir(parents=True)
    for name, value in {"vendor": "0x8086", "class": "0x030000", "device": "0xb0a0"}.items():
        (pci / name).write_text(value)
    (root / "etc").mkdir()
    (root / "etc/os-release").write_text("ID=ubuntu\nVERSION_ID=24.04\nVERSION_CODENAME=noble\n")
    if installed:
        vendors = root / "etc/OpenCL/vendors"
        vendors.mkdir(parents=True)
        (vendors / "intel.icd").write_text("libigdrcl.so\n")
    dri = root / "dev/dri"
    dri.mkdir(parents=True)
    node = dri / "renderD128"
    node.touch(mode=0o600 if accessible else 0o000)
    source = (Path(__file__).resolve().parents[1] / "first_run.sh").read_text()
    for path in ("/sys/bus/pci/devices", "/etc/os-release", "/etc/OpenCL/vendors", "/dev/dri", "/usr/lib"):
        source = source.replace(path, str(root) + path)
    device_listing = "printf 'Platform #0: Intel(R) OpenCL Graphics\\n `-- Device #0: Intel GPU\\n'" if enumerated else ":"
    prelude = f"""
uname() {{ case "$1" in -s) echo Linux ;; -m) echo x86_64 ;; -r) echo 7.0.0 ;; esac; }}
id() {{ case "$1" in -u) echo 1000 ;; -un|-nG) echo fixture-user ;; esac; }}
clinfo() {{ {device_listing}; }}
ldconfig() {{ :; }}
getent() {{ return 1; }}
sudo() {{ echo INSTALL_PRIVILEGE_REQUEST; return 1; }}
usermod() {{ echo UNEXPECTED_USERMOD; return 1; }}
"""
    script = tmp_path / "first_run.sh"
    script.write_text(prelude + source)
    result = subprocess.run(
        ["sh", str(script)], text=True, capture_output=True,
        env={**os.environ, "VIAM_OPENVINO_DRIVERS": mode, "SUDO_USER": "fixture-user"}, timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "UNEXPECTED_USERMOD" not in result.stdout
    return result.stdout + result.stderr


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root bypasses fixture file permissions")
@pytest.mark.parametrize("mode", ["auto", "report"])
def test_installed_gpu_with_denied_render_access_does_not_request_driver_upgrade(tmp_path, mode):
    output = run_gpu_setup(tmp_path, accessible=False, mode=mode)
    assert "permission" in output.lower()
    assert "render" in output
    assert "INSTALL_PRIVILEGE_REQUEST" not in output
    assert "runtime too old" not in output


@pytest.mark.parametrize("installed", [True, False])
def test_accessible_gpu_without_working_runtime_still_requests_install_privileges(tmp_path, installed):
    output = run_gpu_setup(tmp_path, installed=installed)
    assert "INSTALL_PRIVILEGE_REQUEST" in output
    assert "passwordless sudo is unavailable" in output


def test_working_gpu_runtime_does_not_request_install_privileges(tmp_path):
    output = run_gpu_setup(tmp_path, enumerated=True)
    assert "GPU compute runtime installed and lists the GPU" in output
    assert "INSTALL_PRIVILEGE_REQUEST" not in output
