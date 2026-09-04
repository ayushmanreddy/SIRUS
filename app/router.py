from __future__ import annotations

import re
from enum import StrEnum

from app.config import Settings


class TaskType(StrEnum):
    GENERAL = "general"
    CODE = "code"
    VISION = "vision"


CODE_PATTERN = re.compile(
    r"\b(code|python|javascript|typescript|java|debug|bug|script|function|class|api)\b|def\s+",
    re.IGNORECASE,
)


def classify_task(prompt: str, has_image: bool = False) -> TaskType:
    if has_image:
        return TaskType.VISION

    if CODE_PATTERN.search(prompt):
        return TaskType.CODE

    return TaskType.GENERAL


def pick_model(settings: Settings, task: TaskType) -> str:
    model_map = {
        TaskType.GENERAL: settings.reasoning_model,
        TaskType.CODE: settings.coding_model,
        TaskType.VISION: settings.vision_model,
    }
    return model_map[task]