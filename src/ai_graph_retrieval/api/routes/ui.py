"""One static page for asking a question and drawing the answer graph."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(include_in_schema=False)
_PAGE = Path(__file__).resolve().parents[2] / "ui" / "index.html"


@router.get("/ui")
async def ui_page() -> FileResponse:
    return FileResponse(_PAGE, media_type="text/html")
