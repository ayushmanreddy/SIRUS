"""Resource monitoring infrastructure for the Sovereign AI Workbench.

Provides modular monitoring capability for CPU, RAM, GPU, and inference
performance metrics. Designed so the frontend dashboard can consume these
metrics via a standard interface.

Monitoring data is collected snapshotted basis - continuous streaming is
not implied; callers request a snapshot when needed.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("sovereign_workbench.monitor")


@dataclass
class CpuMetrics:
    """CPU utilization metrics."""
    usage_percent: float = 0.0
    cores_total: int = 0
    cores_active: int = 0
    frequency_mhz: Optional[float] = None
    temperature_c: Optional[float] = None


@dataclass
class RamMetrics:
    """RAM utilization metrics."""
    total_mb: int = 0
    used_mb: int = 0
    free_mb: int = 0
    usage_percent: float = 0.0
    swap_total_mb: int = 0
    swap_used_mb: int = 0


@dataclass
class GpuMetrics:
    """GPU utilization metrics."""
    index: int = 0
    name: str = "unknown"
    vendor: str = "unknown"
    memory_total_mb: int = 0
    memory_used_mb: int = 0
    memory_free_mb: int = 0
    utilization_percent: float = 0.0
    temperature_c: Optional[float] = None
    driver_version: str = "unknown"


@dataclass
class ModelPerformanceMetrics:
    """Model inference performance metrics."""
    model_name: str = ""
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    total_requests: int = 0
    error_count: int = 0
    last_request_ms: Optional[float] = None


@dataclass
class MonitoringSnapshot:
    """A complete snapshot of all resource metrics at a point in time."""
    timestamp: float = field(default_factory=time.time)
    cpu: CpuMetrics = field(default_factory=CpuMetrics)
    ram: RamMetrics = field(default_factory=RamMetrics)
    gpu: GpuMetrics = field(default_factory=GpuMetrics)
    model: ModelPerformanceMetrics = field(default_factory=ModelPerformanceMetrics)
    healthy: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "cpu": {
                "usage_percent": self.cpu.usage_percent,
                "cores_total": self.cpu.cores_total,
                "cores_active": self.cpu.cores_active,
                "frequency_mhz": self.cpu.frequency_mhz,
                "temperature_c": self.cpu.temperature_c,
            },
            "ram": {
                "total_mb": self.ram.total_mb,
                "used_mb": self.ram.used_mb,
                "free_mb": self.ram.free_mb,
                "usage_percent": self.ram.usage_percent,
                "swap_total_mb": self.ram.swap_total_mb,
                "swap_used_mb": self.ram.swap_used_mb,
            },
            "gpu": {
                "index": self.gpu.index,
                "name": self.gpu.name,
                "vendor": self.gpu.vendor,
                "memory_total_mb": self.gpu.memory_total_mb,
                "memory_used_mb": self.gpu.memory_used_mb,
                "memory_free_mb": self.gpu.memory_free_mb,
                "utilization_percent": self.gpu.utilization_percent,
                "temperature_c": self.gpu.temperature_c,
                "driver_version": self.gpu.driver_version,
            },
            "model": {
                "model_name": self.model.model_name,
                "avg_latency_ms": self.model.avg_latency_ms,
                "min_latency_ms": self.model.min_latency_ms,
                "max_latency_ms": self.model.max_latency_ms,
                "throughput_tokens_per_sec": self.model.throughput_tokens_per_sec,
                "total_requests": self.model.total_requests,
                "error_count": self.model.error_count,
                "last_request_ms": self.model.last_request_ms,
            },
            "healthy": self.healthy,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MonitoringSnapshot":
        """Create a snapshot from a dictionary (e.g., from JSON deserialization)."""
        ts = data.get("timestamp", time.time())
        cpu_data = data.get("cpu", {})
        ram_data = data.get("ram", {})
        gpu_data = data.get("gpu", {})
        model_data = data.get("model", {})

        cpu = CpuMetrics(
            usage_percent=cpu_data.get("usage_percent", 0.0),
            cores_total=cpu_data.get("cores_total", 0),
            cores_active=cpu_data.get("cores_active", 0),
            frequency_mhz=cpu_data.get("frequency_mhz"),
            temperature_c=cpu_data.get("temperature_c"),
        )

        ram = RamMetrics(
            total_mb=ram_data.get("total_mb", 0),
            used_mb=ram_data.get("used_mb", 0),
            free_mb=ram_data.get("free_mb", 0),
            usage_percent=ram_data.get("usage_percent", 0.0),
            swap_total_mb=ram_data.get("swap_total_mb", 0),
            swap_used_mb=ram_data.get("swap_used_mb", 0),
        )

        gpu = GpuMetrics(
            index=gpu_data.get("index", 0),
            name=gpu_data.get("name", "unknown"),
            vendor=gpu_data.get("vendor", "unknown"),
            memory_total_mb=gpu_data.get("memory_total_mb", 0),
            memory_used_mb=gpu_data.get("memory_used_mb", 0),
            memory_free_mb=gpu_data.get("memory_free_mb", 0),
            utilization_percent=gpu_data.get("utilization_percent", 0.0),
            temperature_c=gpu_data.get("temperature_c"),
            driver_version=gpu_data.get("driver_version", "unknown"),
        )

        model = ModelPerformanceMetrics(
            model_name=model_data.get("model_name", ""),
            avg_latency_ms=model_data.get("avg_latency_ms", 0.0),
            min_latency_ms=model_data.get("min_latency_ms", 0.0),
            max_latency_ms=model_data.get("max_latency_ms", 0.0),
            throughput_tokens_per_sec=model_data.get("throughput_tokens_per_sec", 0.0),
            total_requests=model_data.get("total_requests", 0),
            error_count=model_data.get("error_count", 0),
            last_request_ms=model_data.get("last_request_ms"),
        )

        return cls(
            timestamp=ts,
            cpu=cpu,
            ram=ram,
            gpu=gpu,
            model=model,
            healthy=data.get("healthy", True),
        )