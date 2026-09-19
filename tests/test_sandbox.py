from app.sandbox import run_python


def test_runs_simple_code():
    result = run_python("print(2 + 2)")
    assert "4" in result["stdout"]
    assert result["exit_code"] == 0


def test_captures_errors():
    result = run_python("raise ValueError('boom')")
    assert result["exit_code"] != 0
    assert "ValueError" in result["stderr"]


def test_timeout_is_enforced():
    result = run_python("import time; time.sleep(30)", timeout_seconds=1)
    assert result["timed_out"] is True