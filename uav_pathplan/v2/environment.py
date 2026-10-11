"""Record hardware / software environment for the v2 timing results."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def _cpu_name() -> str:
    try:
        if sys.platform == "win32":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            )
            return out.stdout.strip()
    except Exception:  # noqa: BLE001 - best effort only
        pass
    return platform.processor() or "unknown"


def _ram_bytes() -> int | None:
    try:
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return int(stat.ullTotalPhys)
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except Exception:  # noqa: BLE001 - best effort only
        return None


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        return out.stdout.strip() + (" (dirty working tree)" if dirty.stdout.strip() else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def describe() -> dict:
    """Machine, interpreter, package and threading information."""
    import matplotlib
    import numpy
    import pandas
    import scipy
    import yaml

    ram = _ram_bytes()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cpu": _cpu_name(),
        "logical_cpus": os.cpu_count(),
        "ram_gib": round(ram / 2**30, 1) if ram else None,
        "os": platform.platform(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "packages": {
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "pandas": pandas.__version__,
            "matplotlib": matplotlib.__version__,
            "pyyaml": yaml.__version__,
        },
        "thread_env": {v: os.environ.get(v) for v in THREAD_VARS},
        "git_commit": _git_commit(),
    }
