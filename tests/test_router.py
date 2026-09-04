from app.router import TaskType, classify_task


def test_image_prompt_routes_to_vision():
    assert classify_task("Explain this chart", has_image=True) == TaskType.VISION


def test_coding_prompt_routes_to_code():
    assert classify_task("Debug this Python function") == TaskType.CODE


def test_plain_prompt_routes_to_general():
    assert classify_task("Explain renewable energy") == TaskType.GENERAL