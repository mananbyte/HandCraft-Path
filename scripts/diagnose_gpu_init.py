#!/usr/bin/env python3
"""
GPU initialization diagnostics for PanNuke pipeline.

Usage:
  python scripts/diagnose_gpu_init.py
"""

import os
import platform
import re
import subprocess
import sys
import traceback
from datetime import datetime


def _section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _run_cmd(cmd):
    try:
        out = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return out.returncode, out.stdout.strip()
    except Exception as exc:
        return 1, f"Failed to run {' '.join(cmd)}: {exc}"


def _parse_driver_cuda_version():
    """Parse max CUDA version reported by nvidia-smi (e.g., 12.2)."""
    code, out = _run_cmd(["nvidia-smi"])
    if code != 0:
        return None
    m = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", out)
    return m.group(1) if m else None


def _runtime_int_to_str(runtime_ver):
    """Convert runtime integer (e.g., 12090) to '12.9'."""
    major = runtime_ver // 1000
    minor = (runtime_ver % 1000) // 10
    return f"{major}.{minor}"


def _version_tuple(ver_str):
    try:
        a, b = ver_str.split(".")
        return int(a), int(b)
    except Exception:
        return None


def _print_env():
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER",
        "CUPY_ACCELERATORS",
        "LD_LIBRARY_PATH",
        "CONDA_PREFIX",
    ]
    for k in keys:
        print(f"{k}={os.environ.get(k, '<unset>')}")


def _print_nvidia_smi():
    _section("nvidia-smi summary")
    for cmd in [
        ["nvidia-smi", "-L"],
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,persistence_mode,compute_mode,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
    ]:
        code, out = _run_cmd(cmd)
        print(f"$ {' '.join(cmd)}")
        print(out if out else "<no output>")
        if code != 0:
            print(f"(exit code: {code})")


def _cupy_diagnostics(driver_cuda_version=None):
    _section("CuPy runtime diagnostics")
    try:
        import cupy as cp
        from cupyx.scipy.ndimage import distance_transform_edt as gpu_edt
    except Exception as exc:
        print("CuPy import failed:")
        print(exc)
        return False

    print(f"cupy.__version__ = {cp.__version__}")
    runtime_version_int = None
    runtime_version_str = None
    try:
        runtime_version_int = cp.cuda.runtime.runtimeGetVersion()
        runtime_version_str = _runtime_int_to_str(runtime_version_int)
        print(f"CUDA runtime version = {runtime_version_int} ({runtime_version_str})")
    except Exception as exc:
        print(f"runtimeGetVersion failed: {exc}")

    if driver_cuda_version is not None:
        print(f"Driver max CUDA version = {driver_cuda_version}")

    if runtime_version_str and driver_cuda_version:
        rt = _version_tuple(runtime_version_str)
        drv = _version_tuple(driver_cuda_version)
        if rt and drv and rt > drv:
            print("\n[compatibility warning]")
            print(
                f"CuPy runtime CUDA {runtime_version_str} is newer than driver-supported CUDA {driver_cuda_version}."
            )
            print("This mismatch can cause cudaErrorUnknown during context initialization.")
            print("Fix options:")
            print("  1) Upgrade NVIDIA driver to support CUDA runtime version in this env.")
            print("  2) Recreate env with a CUDA runtime compatible with your current driver.")

    try:
        device_count = cp.cuda.runtime.getDeviceCount()
        print(f"Device count = {device_count}")
    except Exception as exc:
        print(f"getDeviceCount failed: {exc}")
        traceback.print_exc()
        return False

    if device_count == 0:
        print("No CUDA device visible to CuPy.")
        return False

    all_ok = True
    for dev_id in range(device_count):
        print("\n" + "-" * 72)
        print(f"Testing device {dev_id}")
        try:
            with cp.cuda.Device(dev_id):
                # Trigger context creation.
                cur = cp.cuda.runtime.getDevice()
                print(f"Current device = {cur}")

                free_b, total_b = cp.cuda.runtime.memGetInfo()
                print(
                    f"memGetInfo free={free_b/1e9:.2f} GB total={total_b/1e9:.2f} GB"
                )

                # Reset memory pools before allocating.
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                cp.cuda.Device().synchronize()

                # Allocation + arithmetic smoke test.
                arr = cp.zeros((1024, 1024), dtype=cp.float32)
                arr += 1.0
                checksum = float(cp.sum(arr).get())
                print(f"Allocation/arithmetic OK, checksum={checksum:.1f}")

                # EDT test (same primitive used by sampler).
                mask = cp.zeros((256, 256), dtype=cp.float32)
                mask[96:160, 96:160] = 1.0
                dist = gpu_edt(1.0 - mask)
                center = float(dist[128, 128].get())
                print(f"GPU EDT OK, center distance={center:.3f}")

                # Stress small allocations to detect unstable context.
                for _ in range(20):
                    tmp = cp.random.random((512, 512), dtype=cp.float32)
                    _ = float(tmp.mean().get())
                    del tmp
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
                cp.cuda.Device().synchronize()
                print("Repeated allocation test OK")

                del arr, mask, dist
        except Exception as exc:
            all_ok = False
            print(f"Device {dev_id} FAILED: {exc}")
            traceback.print_exc()

    return all_ok


def _explain_common_causes():
    _section("Interpretation hints")
    print("If you see cudaErrorDevicesUnavailable:")
    print("1. Device may be in a bad transient state after a crashed CUDA process.")
    print("2. Driver/runtime mismatch can surface as context-init failures.")
    print("3. Compute mode or container visibility can block context creation.")
    print("4. A stale process may hold exclusive compute mode even with low utilization.")
    print("\nFast recovery checklist:")
    print("- Restart Python/kernel/session and re-run this script.")
    print("- Verify driver + CUDA runtime from nvidia-smi and CuPy output.")
    print("- Ensure CUDA_VISIBLE_DEVICES is not empty/misconfigured.")
    print("- If persistent, reboot GPU host or reset GPU (admin mode).")


def main():
    print("PanNuke GPU initialization diagnostics")
    print(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")

    _section("Environment variables")
    _print_env()

    driver_cuda = _parse_driver_cuda_version()
    _print_nvidia_smi()
    ok = _cupy_diagnostics(driver_cuda_version=driver_cuda)
    _explain_common_causes()

    _section("Final result")
    if ok:
        print("PASS: CuPy context and EDT checks succeeded.")
        sys.exit(0)

    print("FAIL: One or more GPU init checks failed.")
    sys.exit(1)


if __name__ == "__main__":
    main()
