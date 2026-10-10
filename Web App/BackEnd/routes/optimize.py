import logging
from typing import Optional
from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    Header,
    HTTPException,
    Depends,
    status,
)
from fastapi.responses import StreamingResponse

from config import MAX_UPLOAD_SIZE_BYTES, ALLOWED_EXTENSIONS
from services.pipeline_service import stream_pipeline
from utils.pdf import extract_text_from_pdf
from utils.rate_limit import check_rate_limit

logger = logging.getLogger("ai_resume_optimizer.routes.optimize")
router = APIRouter(prefix="/api/optimize", tags=["Optimization"])


@router.post("/stream", dependencies=[Depends(check_rate_limit)])
async def optimize_stream(
    resume: UploadFile = File(..., description="Candidate's resume in PDF format"),
    jd_text: str = Form(..., description="Target Job Description text"),
    x_groq_api_key: Optional[str] = Header(
        None,
        alias="X-Groq-API-Key",
        description="Groq API key (BYOK model)"
    ),
):
    """
    Ingests candidate resume PDF + job description text and streams real-time
    LangGraph execution progress via Server-Sent Events (SSE).
    """
    # 1. Validate JD Text
    clean_jd = (jd_text or "").strip()
    if len(clean_jd) < 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job description text is too short. Please provide a detailed JD (at least 50 characters)."
        )

    # 2. Validate PDF file extension
    filename = (resume.filename or "").lower()
    if not any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Only PDF resumes ({', '.join(ALLOWED_EXTENSIONS)}) are accepted."
        )

    # 3. Read & Validate File Size
    contents = await resume.read()
    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Resume file size exceeds maximum limit of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB."
        )

    # 4. Extract Text from PDF
    try:
        resume_text = extract_text_from_pdf(contents)
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not extract text from the PDF file: {str(e)}"
        )

    if len(resume_text) < 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The extracted resume text is too brief or empty. Please check your PDF."
        )

    # 5. Return SSE Streaming Response
    generator = stream_pipeline(
        resume_text=resume_text,
        jd_text=clean_jd,
        api_key=x_groq_api_key
    )

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
