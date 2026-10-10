import json
import logging
from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage

from core.models import (
    ResumeData,
    JDData,
    OptimizedResume,
    ATSScore,
    CriticOutput,
    ValidatorOutput,
    FinalReport,
    GraphState,
)
from core.llm import get_llm_with_fallback
from core.parser import parse_json_from_response

logger = logging.getLogger("ai_resume_optimizer.agents")


# ─── AGENT 0: Document Validator ─────────────────────────────────────────────
def validator_agent(state: GraphState) -> dict:
    """Guardrail node to check if the uploaded document is a valid resume."""
    resume_text = state.get("resume_text", "")
    api_key = state.get("api_key")
    llm = get_llm_with_fallback("low", api_key=api_key)
    
    prompt = f"Analyze this text and determine if it appears to be a resume/CV. Respond in JSON. Text:\n{resume_text[:2000]}"
    try:
        response = llm.with_retry(stop_after_attempt=3).invoke([
            SystemMessage(content="""Return valid JSON: {"is_resume": true/false, "reason": "..."}"""),
            HumanMessage(content=prompt)
        ])
        parsed = parse_json_from_response(response.content)
        val = ValidatorOutput(**parsed)
        logger.info(f"Validation: {'Passed' if val.is_resume else 'Failed'} - {val.reason}")
        return {"is_valid_resume": val.is_resume}
    except Exception as e:
        logger.warning(f"Validation error, allowing to proceed: {e}")
        return {"is_valid_resume": True}


# ─── AGENT 1: Resume Parser ──────────────────────────────────────────────────
def resume_parser_agent(state: GraphState) -> dict:
    """Parse resume text into structured ResumeData."""
    resume_text = state["resume_text"]
    api_key = state.get("api_key")
    
    system_prompt = """You are an expert resume parser. Extract structured information from the resume text.
Return ONLY a valid JSON object matching this exact schema (no extra text):

{
  "skills": ["skill1", "skill2", ...],
  "projects": [
    {
      "name": "Project Name",
      "description": "Brief description",
      "technologies": ["tech1", "tech2"],
      "highlights": ["Key achievement or outcome"]
    }
  ],
  "experience": [
    {
      "title": "Job Title",
      "company": "Company Name",
      "duration": "Start – End",
      "bullet_points": ["Responsibility or achievement"]
    }
  ],
  "education": [
    {
      "degree": "Degree Name",
      "institution": "University Name",
      "year": "Graduation Year",
      "gpa": "GPA or null"
    }
  ],
  "certifications": ["cert1", "cert2"]
}

Rules:
- Extract ONLY information explicitly stated in the resume.
- Do NOT invent or hallucinate any data.
- If a section is empty, use an empty list [].
- List ONLY the top 10 most important skills to save space. Do NOT exceed 10 skills!"""

    llm = get_llm_with_fallback("low", api_key=api_key)
    response = llm.invoke([
        SystemMessage(content=system_prompt), 
        HumanMessage(content=f"Parse the following resume:\n\n{resume_text}")
    ])
    
    parsed = parse_json_from_response(response.content)
    resume_data = ResumeData(**parsed)
    
    logger.info(
        f"Resume parsed: {len(resume_data.skills)} skills, "
        f"{len(resume_data.projects)} projects, "
        f"{len(resume_data.experience)} experience entries."
    )
    return {"resume_data": resume_data.model_dump()}


# ─── AGENT 2: Job Description Parser ─────────────────────────────────────────
def jd_parser_agent(state: GraphState) -> dict:
    """Parse job description text into structured JDData."""
    jd_text = state["jd_text"]
    api_key = state.get("api_key")
    
    system_prompt = """You are an expert job description analyzer. Extract structured information from the JD.
Return ONLY a valid JSON object matching this exact schema (no extra text):

{
  "required_skills": ["skill1", "skill2", ...],
  "preferred_skills": ["skill1", "skill2", ...],
  "responsibilities": ["responsibility1", "responsibility2", ...],
  "experience_requirements": "e.g., 2-4 years in ML/AI"
}

Rules:
- Extract ALL required and preferred skills mentioned.
- Normalize skill names (e.g., 'Python 3' → 'Python').
- Capture the full experience requirement string.
- Include both technical and domain-specific skills."""

    llm = get_llm_with_fallback("low", api_key=api_key)
    response = llm.invoke([
        SystemMessage(content=system_prompt), 
        HumanMessage(content=f"Parse the following job description:\n\n{jd_text}")
    ])
    
    parsed = parse_json_from_response(response.content)
    jd_data = JDData(**parsed)
    
    logger.info(
        f"JD parsed: {len(jd_data.required_skills)} required skills, "
        f"{len(jd_data.preferred_skills)} preferred skills."
    )
    return {"jd_data": jd_data.model_dump()}


# ─── AGENT 3 & 6: ATS Scoring Engine ─────────────────────────────────────────
def ats_scoring_agent(state: GraphState, score_key: str = "ats_score_before") -> dict:
    """Score the resume against the JD across 5 dimensions."""
    resume_data = state["resume_data"]
    jd_data = state["jd_data"]
    resume_text = state["resume_text"]
    api_key = state.get("api_key")
    
    optimized_context = ""
    if score_key == "ats_score_after" and state.get("optimized_resume"):
        optimized = state["optimized_resume"]
        enhanced_text = "\n".join([
            f"- {b.get('enhanced', '')}" for b in optimized.get("enhanced_bullets", [])
        ])
        optimized_context = f"""\n\n=== OPTIMIZED RESUME BULLETS (use these for scoring) ===
{enhanced_text}
\nScore the resume AS IF these enhanced bullet points replaced the originals."""
    
    system_prompt = """You are an expert ATS (Applicant Tracking System) scoring engine.
Evaluate the resume against the job description strictly on these five distinct dimensions:

1. skill_match (0-100): Percentage of required technical tools, libraries, and frameworks present.
2. keyword_coverage (0-100): Alignment of specific domain phrases (e.g., CI/CD, Agile) and methodology terms.
3. experience_alignment (0-100): Does the candidate's total years of experience and job titles match 
   the seniority demanded by the JD? (e.g., penalize freshers applying for senior roles heavily).
4. resume_quality (0-100): Use of action verbs, quantified metrics (percentages, numbers), and STAR/XYZ formatting.
5. project_relevance (0-100): Practical alignment of listed projects/work history to primary JD daily tasks.

Scoring Rules & Guardrails:
- Hard Knock-Out Check: If the job description explicitly requires a minimum number of years of experience 
  and the candidate has 0 years or is a fresher, cap 'experience_alignment' at 0.
- Do not let high keyword density mask a complete deficit in real-world professional tenure.

Compute the final overall_score using this industry-weighted formula:
  overall_score = (skill_match * 0.25) + (keyword_coverage * 0.15) + (experience_alignment * 0.25) + (resume_quality * 0.15) + (project_relevance * 0.20)

Return ONLY a valid JSON object:
{
  "overall_score": <float>,
  "is_knocked_out": <bool>,
  "breakdown": {
    "skill_match": <float>,
    "keyword_coverage": <float>,
    "experience_alignment": <float>,
    "resume_quality": <float>,
    "project_relevance": <float>
  },
  "reasoning": "Detailed breakdown of strengths and gaps."
}"""

    human_message = f"""=== RESUME DATA ===
{json.dumps(resume_data, indent=2)} 

=== RESUME FULL TEXT ===
{resume_text}

=== JOB DESCRIPTION DATA ===
{json.dumps(jd_data, indent=2)}{optimized_context}"""

    llm = get_llm_with_fallback("low", api_key=api_key)
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_message)
    ])
    
    parsed = parse_json_from_response(response.content)
    ats_score = ATSScore(**parsed)
    
    label = "BEFORE" if score_key == "ats_score_before" else "AFTER"
    logger.info(f"ATS Score ({label}): {ats_score.overall_score:.1f}/100")
    return {score_key: ats_score.model_dump()}


def ats_score_before_node(state: GraphState) -> dict:
    return ats_scoring_agent(state, score_key="ats_score_before")


def ats_score_after_node(state: GraphState) -> dict:
    return ats_scoring_agent(state, score_key="ats_score_after")


# ─── AGENT 4: Gap Analysis ───────────────────────────────────────────────────
def gap_analysis_agent(state: GraphState) -> dict:
    """Deterministic skill matching + LLM analysis of weak areas."""
    resume_data = state["resume_data"]
    jd_data = state["jd_data"]
    api_key = state.get("api_key")
    
    # Deterministic set matching
    res_skills_lower = {s.lower().strip() for s in resume_data.get("skills", [])}
    jd_skills_lower = {s.lower().strip() for s in jd_data.get("required_skills", []) + jd_data.get("preferred_skills", [])}
    
    matched = list(jd_skills_lower.intersection(res_skills_lower))
    missing = list(jd_skills_lower - res_skills_lower)
    
    system_prompt = """You are a career advisor. Identify weak areas in the candidate's resume (e.g. formatting, missing metrics, poor phrasing).
Return valid JSON: {"weak_areas": ["issue 1", "issue 2"]}"""
    human_message = f"Resume Data:\n{json.dumps(resume_data, indent=2)}\nJD Data:\n{json.dumps(jd_data, indent=2)}"
    
    llm = get_llm_with_fallback("high", api_key=api_key)
    try:
        response = llm.with_retry(stop_after_attempt=3).invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message)
        ])
        parsed = parse_json_from_response(response.content)
        weak_areas = parsed.get("weak_areas", [])
    except Exception:
        weak_areas = ["Consider adding more quantified metrics to bullet points."]
        
    gap_result = {
        "matched_skills": matched,
        "missing_skills": missing,
        "missing_keywords": missing,
        "weak_areas": weak_areas
    }
    logger.info(f"Gap Analysis: {len(matched)} matched, {len(missing)} missing skills.")
    return {"gap_analysis": gap_result}


# ─── AGENT 5: Resume Enhancement ─────────────────────────────────────────────
def resume_enhancement_agent(state: GraphState) -> dict:
    """Enhance resume bullets incorporating missing skills using Critic feedback."""
    resume_data = state["resume_data"]
    jd_data = state["jd_data"]
    gap_analysis = state["gap_analysis"]
    critic_feedback = state.get("critic_feedback", "")
    api_key = state.get("api_key")
    
    feedback_context = f"\nPREVIOUS CRITIC FEEDBACK TO FIX: {critic_feedback}" if critic_feedback and critic_feedback != "PASS" else ""
    
    system_prompt = f"""You are an elite ATS resume writer. Optimize the bullet points by weaving in the following EXACT missing skills logically and honestly.
MISSING SKILLS TO WEAVE IN: {gap_analysis.get('missing_skills', [])}
{feedback_context}

Return ONLY valid JSON matching this schema:
{{
  "enhanced_bullets": [
    {{"original": "original text", "enhanced": "enhanced text", "section": "experience"}}
  ],
  "keywords_incorporated": ["skill1", "skill2"],
  "optimization_summary": "Summary of changes"
}}"""
    human_message = f"Resume Data:\n{json.dumps(resume_data, indent=2)}\nJD Data:\n{json.dumps(jd_data, indent=2)}"
    
    llm = get_llm_with_fallback("high", api_key=api_key)
    try:
        response = llm.with_retry(stop_after_attempt=3).invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message)
        ])
        parsed = parse_json_from_response(response.content)
        optimized = OptimizedResume(**parsed)
        logger.info(f"Enhancements generated: {len(optimized.enhanced_bullets)} bullets updated.")
        return {"optimized_resume": optimized.model_dump()}
    except Exception as e:
        logger.error(f"Enhancement Error: {e}")
        return {"optimized_resume": {"enhanced_bullets": [], "keywords_incorporated": [], "optimization_summary": "Failed to enhance"}}


# ─── AGENT 5.5: Critic Agent ─────────────────────────────────────────────────
def critic_agent(state: GraphState) -> dict:
    """Evaluates the enhanced resume bullets and forces reflection if needed."""
    optimized_resume = state.get("optimized_resume", {})
    gap_analysis = state.get("gap_analysis", {})
    iterations = state.get("enhancement_iterations", 0) + 1
    api_key = state.get("api_key")
    
    if iterations >= 3:
        logger.info("Critic: Max iterations reached. Proceeding.")
        return {"enhancement_iterations": iterations, "critic_feedback": "PASS"}

    llm = get_llm_with_fallback("high", api_key=api_key)
    system_prompt = """You are an expert Resume Editor acting as a Critic. 
Review the enhanced bullets against the original Job Description and missing skills.
Criteria to FAIL (passes_standard=false):
1. LLM Hallucinated fake metrics or fake jobs.
2. The skills were awkwardly stuffed into the bullets (keyword stuffing).
3. The bullets sound like an AI wrote them.

Return JSON: {"passes_standard": true/false, "feedback": "Detailed reason"}"""
    prompt = f"Missing Skills to Add: {gap_analysis.get('missing_skills')}\nEnhanced Resume:\n{json.dumps(optimized_resume, indent=2)}"
    
    try:
        response = llm.with_retry(stop_after_attempt=3).invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ])
        parsed = parse_json_from_response(response.content)
        critic = CriticOutput(**parsed)
        logger.info(f"Critic Iteration {iterations}: {'PASSED' if critic.passes_standard else 'FAILED'}")
        return {
            "enhancement_iterations": iterations,
            "critic_feedback": "PASS" if critic.passes_standard else critic.feedback
        }
    except Exception as e:
        logger.warning(f"Critic failed, bypassing: {e}")
        return {"enhancement_iterations": iterations, "critic_feedback": "PASS"}


# ─── AGENT 7: Report Generator ───────────────────────────────────────────────
def report_generator_agent(state: GraphState) -> dict:
    """Generate the final comprehensive optimization report."""
    ats_before = state["ats_score_before"]
    ats_after = state["ats_score_after"]
    gap = state["gap_analysis"]
    optimized = state["optimized_resume"]
    api_key = state.get("api_key")
    
    score_before = ats_before.get("overall_score", 0.0)
    score_after = ats_after.get("overall_score", 0.0)
    improvement = ((score_after - score_before) / max(score_before, 1)) * 100
    
    system_prompt = """You are an expert career advisor generating a final resume optimization report.

Return ONLY a valid JSON object:
{
  "ats_score_before": <float>,
  "ats_score_after": <float>,
  "improvement_percentage": <float>,
  "skill_match_analysis": {
    "matched": ["skill1", "skill2"],
    "missing": ["skill3", "skill4"]
  },
  "missing_keywords": ["keyword1", "keyword2"],
  "enhanced_bullets": [
    {"original": "...", "enhanced": "...", "section": "..."}
  ],
  "recommendations": [
    "Specific, actionable recommendation 1",
    "Specific, actionable recommendation 2"
  ],
  "executive_summary": "A 2-3 sentence high-level summary of the analysis and key takeaways."
}

Provide at least 5 specific, actionable recommendations.
The executive_summary should be professional and encouraging."""

    human_message = f"""Compile the final report from these inputs:

=== ATS SCORE BEFORE ===
{json.dumps(ats_before, indent=2)}

=== ATS SCORE AFTER ===
{json.dumps(ats_after, indent=2)}

=== IMPROVEMENT ===
Score Before: {score_before:.1f}
Score After: {score_after:.1f}
Improvement: {improvement:.1f}%

=== GAP ANALYSIS ===
{json.dumps(gap, indent=2)}

=== OPTIMIZED RESUME ===
{json.dumps(optimized, indent=2)}"""

    llm = get_llm_with_fallback("high", api_key=api_key)
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_message)
    ])
    
    parsed = parse_json_from_response(response.content)
    report = FinalReport(**parsed)
    logger.info(f"Final Report Generated! Score: {report.ats_score_before:.1f} -> {report.ats_score_after:.1f}")
    return {"final_report": report.model_dump()}


# ─── Routing Logic ────────────────────────────────────────────────────────────
def routing_logic(state: GraphState) -> str:
    if not state.get("is_valid_resume", True):
        return "invalid"
    return "valid"


def reflection_routing(state: GraphState) -> str:
    if state.get("critic_feedback") == "PASS":
        return "pass"
    return "fail"
