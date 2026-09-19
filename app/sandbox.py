from __future__ import annotations

import subprocess
import tempfile
import sys


def run_python(code: str, timeout_seconds: float = 10.0) -> dict:
    """Run Python code in a separate process with isolation.

    - Uses subprocess.run() for process-level isolation
    - Runs in Python isolated mode (-I) to ignore user site-packages and PYTHONPATH
    - Executes in a fresh temp directory so file writes are contained
    - Enforces a wall-clock timeout; on timeout the process is killed
    - Captures stdout and stderr separately

    This is a demo-level isolation boundary (separate process + timeout +
    isolated mode + temp cwd), not a production security sandbox.
    A real deployment would need Docker, gVisor, or similar.
    """
    cwd = tempfile.mkdtemp()
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=cwd,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": "",
            "exit_code": None,
            "timed_out": True,
        }
    finally:
        import shutil

        shutil.rmtree(cwd, ignore_errors=True)