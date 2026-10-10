from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class UserEditBullet(BaseModel):
    original: str
    enhanced: str
    section: str = "experience"


class UserEdits(BaseModel):
    enhanced_bullets: Optional[List[Dict[str, Any]]] = None
    keywords_incorporated: Optional[List[str]] = None


class GenerateRequest(BaseModel):
    """Payload for POST /api/generate (Approve & Export)."""
    resume_data: Dict[str, Any]
    optimized_resume: Dict[str, Any]
    gap_analysis: Optional[Dict[str, Any]] = Field(default_factory=dict)
    user_edits: Optional[UserEdits] = None
