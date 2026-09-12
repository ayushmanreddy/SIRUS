"""Presenter-safe local hardware capability detection."""
from __future__ import annotations
import json
import logging
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger("sovereign_workbench.hardware")


@dataclass(frozen=True)
class GpuInfo:
    name: str
    memory_total_mb: int | None
    memory_used_mb: int | None
    vendor: str = "unknown"


@dataclass(frozen=True)
class CpuInfo:
    cores: int
    model_name: str | None = None
    architecture: str | None = None


@dataclass(frozen=True)
class HardwareSnapshot:
    operating_system: str
    architecture: str
    cpu: CpuInfo
    gpus: list[GpuInfo]
    gpu_detection: str
    ram_total_mb: int
    def safe_dict(self) -> dict[str, object]:
        base = asdict(self)
        # Convert nested dataclasses to dicts recursively
        return json.loads(json.dumps(base, default=str))


def detect_hardware() -> HardwareSnapshot:
    """Detect hardware capabilities for the local AI workbench."""
    gpus: list[GpuInfo] = []
    cpu_info = CpuInfo(cores=os.cpu_count() or 1, architecture=platform.machine())
    cpu_model = _get_cpu_model()
    cpu_arch = platform.machine()

    # Detect GPU
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        if result.returncode == 0:
            for row in result.stdout.splitlines():
                values = [value.strip() for value in row.split(",")]
                if len(values) == 3:
                    try:
                        total, used = int(values[1]), int(values[2])
                    except ValueError:
                        total, used = None, None
                    gpus.append(GpuInfo(values[0], total, used, vendor="NVIDIA"))
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    # If no NVIDIA GPU detected, check for AMD ROCm
    if not gpus:
        try:
            result = subprocess.run(
                ["rocm-smi", "--query-gpu=name,memory_total,memory_used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3, check=False,
            )
            if result.returncode == 0:
                for row in result.stdout.splitlines():
                    values = [value.strip() for value in row.split(",")]
                    if len(values) == 3:
                        try:
                            total, used = int(values[1]), int(values[2])
                        except ValueError:
                            total, used = None, None
                        gpus.append(GpuInfo(values[0], total, used, vendor="AMD"))
        except (FileNotFoundError, subprocess.SubprocessError):
            pass

    # If no GPU detected at all, note CPU fallback
    if not gpus:
        logger.info("No GPU detected; CPU fallback will be used for inference")

    gpu_detection = (
        "NVIDIA GPU capability detected"
        if any(g.vendor == "NVIDIA" for g in gpus)
        else ("AMD GPU capability detected" if any(g.vendor == "AMD" for g in gpus) else "no compatible GPU runtime detected; CPU fallback may be required")
    )

    # Detect total RAM
    ram_total_mb = _get_total_ram_mb()

    return HardwareSnapshot(
        operating_system=platform.system(),
        architecture=cpu_arch,
        cpu=cpu_info,
        gpus=gpus,
        gpu_detection=gpu_detection,
        ram_total_mb=ram_total_mb,
    )


def _get_cpu_model() -> str | None:
    """Try to detect CPU model name."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line.lower():
                    # Get the first model name found
                    model = line.split(":", 1)[1].strip() if ":" in line else line.strip()
                    return model[:60] if model else None
    except (FileNotFoundError, OSError):
        pass
    return None


def _get_total_ram_mb() -> int:
    """Get total system RAM in MB."""
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    # kB to MB
                    return int(line.split()[1]) // 1024
    except (FileNotFoundError, ValueError):
        pass
    # Fallback: estimate from Python
    import os
    try:
        return int(os.sysconf("SC_PHYSICAL_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 * 1024))
    except (AttributeError, ValueError):
        return 8192  # safe default