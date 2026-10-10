from typing import List, Dict, Any, Optional, TypedDict
from pydantic import BaseModel, Field


class ResumeData(BaseModel):
    skills: List[str] = Field(default_factory=list)
    projects: List[Dict[str, Any]] = Field(default_factory=list)
    experience: List[Dict[str, Any]] = Field(default_factory=list)
    education: List[Dict[str, Any]] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)


class JDData(BaseModel):
    required_skills: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    responsibilities: List[str] = Field(default_factory=list)
    qualifications: List[str] = Field(default_factory=list)
    key_requirements: List[str] = Field(default_factory=list)
    experience_requirements: Optional[str] = None


class EnhancedBullet(BaseModel):
    original: str = ""
    enhanced: str = ""
    section: str = "experience"


class OptimizedResume(BaseModel):
    enhanced_bullets: List[Dict[str, Any]] = Field(default_factory=list)
    keywords_incorporated: List[str] = Field(default_factory=list)
    optimization_summary: str = ""


class ATSBreakdown(BaseModel):
    skill_match: float = 0.0
    keyword_coverage: float = 0.0
    experience_alignment: float = 0.0
    resume_quality: float = 0.0
    project_relevance: float = 0.0


class ATSScore(BaseModel):
    overall_score: float = 0.0
    is_knocked_out: bool = False
    breakdown: ATSBreakdown = Field(default_factory=ATSBreakdown)
    reasoning: str = ""


class CriticOutput(BaseModel):
    passes_standard: bool = Field(default=False, description="True if enhancement is high quality, False if it hallucinates or sounds robotic")
    feedback: str = Field(default="", description="Feedback for the enhancement node to improve")


class ValidatorOutput(BaseModel):
    is_resume: bool = Field(default=True, description="True if document is a resume, False otherwise")
    reason: str = Field(default="", description="Reason for decision")


class FinalReport(BaseModel):
    ats_score_before: float = 0.0
    ats_score_after: float = 0.0
    improvement_percentage: float = 0.0
    skill_match_analysis: Dict[str, Any] = Field(default_factory=dict)
    missing_keywords: List[str] = Field(default_factory=list)
    enhanced_bullets: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    executive_summary: str = ""


class GraphState(TypedDict, total=False):
    """State object passed between LangGraph nodes."""
    resume_text: str
    jd_text: str
    resume_data: dict
    jd_data: dict
    ats_score_before: dict
    gap_analysis: dict
    optimized_resume: dict
    ats_score_after: dict
    final_report: dict
    optimized_latex: str
    is_valid_resume: bool
    enhancement_iterations: int
    critic_feedback: str
    api_key: str  # Injected per-session Groq API key
