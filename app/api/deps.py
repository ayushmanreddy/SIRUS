from __future__ import annotations

from fastapi import Request

from app.config import Settings
from app.dependencies import AppContainer


def get_container(request: Request) -> AppContainer:
    return request.app.state.container


def get_settings(request: Request) -> Settings:
    return request.app.state.container.settings
