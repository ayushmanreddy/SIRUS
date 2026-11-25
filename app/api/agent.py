from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.api.deps import get_container
from app.llm_client import ModelUnavailableError
from app.router import TaskType, classify_task, pick_model
from app.sandbox import run_python

router = APIRouter(tags=["agent"])


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=12000)
    task_type: TaskType | None = None
    has_image: bool = False


class GenerateResponse(BaseModel):
    task_type: str
    model: str
    response: str


class CodeExecutionRequest(BaseModel):
    code: str = Field(max_length=5000)


class CodeExecutionResponse(BaseModel):
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool


@router.post("/agent/generate", response_model=GenerateResponse)
async def generate_agent_response(request: Request, payload: GenerateRequest) -> GenerateResponse:
    container = get_container(request)
    task = payload.task_type or classify_task(payload.prompt, has_image=payload.has_image)
    model = pick_model(container.settings, task)
    correlation_id = request.state.correlation_id

    try:
        generated_text = await container.llm_client.generate(model, payload.prompt)
    except ModelUnavailableError:
        container.audit.record(
            "agent.generation_failed",
            correlation_id,
            {"task_type": task.value, "model": model, "reason": "model_unavailable"},
        )
        raise

    container.audit.record(
        "agent.generation_succeeded",
        correlation_id,
        {"task_type": task.value, "model": model},
    )
    return GenerateResponse(task_type=task.value, model=model, response=generated_text)


@router.post("/agent/execute_code", response_model=CodeExecutionResponse)
async def execute_code(request: Request, payload: CodeExecutionRequest) -> CodeExecutionResponse:
    container = get_container(request)
    result = run_python(payload.code)
    container.audit.record(
        "code_execution",
        request.state.correlation_id,
        {
            "task_type": "code_execution",
            "exit_code": result["exit_code"],
            "timed_out": result["timed_out"],
        },
    )
    return CodeExecutionResponse(
        stdout=result["stdout"],
        stderr=result["stderr"],
        exit_code=result["exit_code"],
        timed_out=result["timed_out"],
    )
