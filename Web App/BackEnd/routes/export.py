import os
import re
import logging
from typing import Optional
from pathlib import Path
from fastapi import (
    APIRouter,
    Header,
    HTTPException,
    status,
)
from fastapi.responses import FileResponse

from config import TEMP_DIR
from schemas.requests import GenerateRequest
from schemas.responses import GenerateResponse, HealthResponse
from services.export_service import generate_all_exports
from utils.latex_compiler import is_latex_compiler_available

logger = logging.getLogger("ai_resume_optimizer.routes.export")
router = APIRouter(prefix="/api", tags=["Export & Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint verifying system readiness and LaTeX compiler status."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        latex_compiler_available=is_latex_compiler_available()
    )


@router.post("/generate", response_model=GenerateResponse)
async def generate_final_documents(
    payload: GenerateRequest,
    x_groq_api_key: Optional[str] = Header(
        None,
        alias="X-Groq-API-Key",
        description="Groq API key (BYOK model)"
    ),
):
    """
    Applies human review edits, weaves bullets into LaTeX template, compiles PDF,
    and creates downloadable DOCX, TEX, and PDF files.
    """
    try:
        export_result = generate_all_exports(
            resume_data=payload.resume_data,
            optimized_resume=payload.optimized_resume,
            user_edits=payload.user_edits.model_dump() if payload.user_edits else None,
            api_key=x_groq_api_key,
        )
        return GenerateResponse(**export_result)
    except Exception as e:
        logger.exception("Document generation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate final documents: {str(e)}"
        )


@router.get("/download/{session_id}/{filename}")
async def download_file(session_id: str, filename: str):
    """
    Safely downloads a generated file (PDF, DOCX, or TEX).
    Guards against path traversal attacks.
    """
    # 1. Sanitize parameters
    if not re.match(r"^[a-zA-Z0-9_-]+$", session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID format.")

    clean_filename = os.path.basename(filename)
    allowed_suffixes = {".pdf", ".docx", ".tex"}
    suffix = Path(clean_filename).suffix.lower()

    if suffix not in allowed_suffixes:
        raise HTTPException(status_code=400, detail="Requested file type is not downloadable.")

    file_path = TEMP_DIR / session_id / clean_filename

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="The requested file was not found or has expired."
        )

    # Determine media type
    media_types = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".tex": "application/x-tex",
    }
    media_type = media_types.get(suffix, "application/octet-stream")

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=clean_filename,
    )
