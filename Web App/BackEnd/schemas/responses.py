import time
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    latex_compiler_available: bool = False
    timestamp: float = Field(default_factory=time.time)


class GenerateResponse(BaseModel):
    session_id: str
    pdf_url: Optional[str] = None
    tex_url: Optional[str] = None
    docx_url: Optional[str] = None
    message: Optional[str] = None


class SSEStatusEvent(BaseModel):
    agent: str
    message: str
    progress: int


class SSEResultEvent(BaseModel):
    ats_score_before: Dict[str, Any]
    ats_score_after: Dict[str, Any]
    gap_analysis: Dict[str, Any]
    optimized_resume: Dict[str, Any]
    final_report: Dict[str, Any]


class SSEDoneEvent(BaseModel):
    message: str = "Pipeline complete"
    elapsed_seconds: float


class SSEErrorEvent(BaseModel):
    message: str
    code: str
