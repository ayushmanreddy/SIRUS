"""Presenter-safe local hardware capability detection."""
from __future__ import annotations
import os
import platform
import subprocess
from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class GpuInfo:
    name: str
    memory_total_mb: int | None
    memory_used_mb: int | None

@dataclass(frozen=True)
class HardwareSnapshot:
    operating_system: str
    architecture: str
    cpu_cores: int | None
    gpus: list[GpuInfo]
    gpu_detection: str
    def safe_dict(self) -> dict[str, object]:
        return asdict(self)

def detect_hardware() -> HardwareSnapshot:
    gpus: list[GpuInfo] = []
    detection = "no compatible GPU runtime detected; CPU fallback may be required"
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3, check=False)
        if result.returncode == 0:
            for row in result.stdout.splitlines():
                values = [value.strip() for value in row.split(",")]
                if len(values) == 3:
                    try: total, used = int(values[1]), int(values[2])
                    except ValueError: total, used = None, None
                    gpus.append(GpuInfo(values[0], total, used))
            if gpus: detection = "NVIDIA GPU capability detected"
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return HardwareSnapshot(platform.system(), platform.machine(), os.cpu_count(), gpus, detection)
