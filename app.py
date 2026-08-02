import os
import json
import re
import textwrap
import time
import subprocess
import shutil
from typing import List, Optional, Dict, Any, TypedDict

import fitz  # PyMuPDF for PDF extraction
import streamlit as st
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set premium Streamlit page configuration
st.set_page_config(
    page_title="AI Resume Optimizer & ATS Builder",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Premium CSS Styling (Dark/Light mode compliant, custom metric cards, tables, and buttons)
st.markdown("""
<style>
    /* Google Font Import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #0e1117;
    }
    
    /* Premium Metric Card */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px 25px;
        border-radius: 16px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease-in-out;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        border-color: #ff4b4b;
        box-shadow: 0 10px 30px rgba(255, 75, 75, 0.15);
    }
    
    /* Tab Styling */
    button[data-baseweb="tab"] {
        font-size: 16px !important;
        font-weight: 600 !important;
        color: #8a8e99 !important;
        border-bottom-width: 2px !important;
        transition: all 0.3s ease !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ff4b4b !important;
        border-bottom-color: #ff4b4b !important;
    }
    
    /* Enhancements comparison table */
    .comparison-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 15px;
        border-radius: 8px;
        overflow: hidden;
    }
    
    .comparison-table th {
        background-color: #1e222b;
        color: #ffffff;
        font-weight: 600;
        padding: 12px 15px;
        text-align: left;
        border: 1px solid #2d3139;
    }
    
    .comparison-table td {
        padding: 15px;
        border: 1px solid #2d3139;
        vertical-align: top;
        line-height: 1.5;
    }
    
    .original-bullet {
        background-color: rgba(255, 75, 75, 0.05);
        color: #ff7676;
    }
    
    .enhanced-bullet {
        background-color: rgba(76, 175, 80, 0.05);
        color: #7cd980;
        font-weight: 500;
    }
    
    /* Section Headers */
    .section-title {
        font-size: 24px;
        font-weight: 700;
        border-bottom: 2px solid #ff4b4b;
        padding-bottom: 8px;
        margin-top: 25px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DEFINE PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────────────────────

class ProjectDetail(BaseModel):
    name: str = Field(description="Project name or title")
    description: str = Field(description="Brief description of the project")
    technologies: List[str] = Field(default_factory=list, description="Technologies/tools used")
    highlights: List[str] = Field(default_factory=list, description="Key achievements or outcomes")

class ExperienceDetail(BaseModel):
    title: str = Field(description="Job title")
    company: str = Field(description="Company name")
    duration: str = Field(description="Employment period, e.g. 'Jan 2022 – Dec 2023'")
    bullet_points: List[str] = Field(default_factory=list, description="Responsibility/achievement bullet points")

class EducationDetail(BaseModel):
    degree: str = Field(description="Degree name, e.g. 'B.Tech in Computer Science'")
    institution: str = Field(description="University or school name")
    year: str = Field(default="", description="Graduation year or period")
    gpa: Optional[str] = Field(default=None, description="GPA or percentage if mentioned")

class ResumeData(BaseModel):
    skills: List[str] = Field(default_factory=list, description="Technical and soft skills")
    projects: List[ProjectDetail] = Field(default_factory=list, description="Projects")
    experience: List[ExperienceDetail] = Field(default_factory=list, description="Work experience entries")
    education: List[EducationDetail] = Field(default_factory=list, description="Education entries")
    certifications: List[str] = Field(default_factory=list, description="Certifications or courses")

class JDData(BaseModel):
    required_skills: List[str] = Field(default_factory=list, description="Must-have skills")
    preferred_skills: List[str] = Field(default_factory=list, description="Nice-to-have skills")
    responsibilities: List[str] = Field(default_factory=list, description="Key responsibilities")
    experience_requirements: str = Field(default="", description="Experience level required, e.g. '3+ years'")

class ScoreBreakdown(BaseModel):
    skill_match: float = Field(description="Score for skill overlap (0-100)")
    experience_alignment: float = Field(description="Score for experience seniority matching (0-100)")
    keyword_coverage: float = Field(description="Score for JD keyword presence in resume (0-100)")
    resume_quality: float = Field(description="Score for bullet-point clarity, action verbs, quantification (0-100)")
    project_relevance: float = Field(description="Score for project alignment to JD tasks (0-100)")

class ATSScore(BaseModel):
    overall_score: float = Field(description="Weighted overall ATS score (0-100)")
    breakdown: ScoreBreakdown = Field(description="Category-wise score breakdown")
    reasoning: str = Field(default="", description="Brief explanation of the scoring")

class GapAnalysis(BaseModel):
    matched_skills: List[str] = Field(default_factory=list, description="Skills present in both resume and JD")
    missing_skills: List[str] = Field(default_factory=list, description="JD skills not found in the resume")
    missing_keywords: List[str] = Field(default_factory=list, description="Important JD keywords absent from resume")
    weak_areas: List[str] = Field(default_factory=list, description="Resume sections that need improvement")

class EnhancedBullet(BaseModel):
    original: str = Field(description="The original bullet point from the resume")
    enhanced: str = Field(description="Improved version using STAR/XYZ format")
    section: str = Field(description="Which section this belongs to (experience, projects, etc.)")

class OptimizedResume(BaseModel):
    enhanced_bullets: List[EnhancedBullet] = Field(default_factory=list, description="Improved bullet points")
    keywords_incorporated: List[str] = Field(default_factory=list, description="JD keywords woven into the enhanced text")
    optimization_summary: str = Field(default="", description="Summary of changes made")

class FinalReport(BaseModel):
    ats_score_before: float = Field(description="ATS score before optimization")
    ats_score_after: float = Field(description="ATS score after optimization")
    improvement_percentage: float = Field(description="Percentage improvement")
    skill_match_analysis: Dict[str, List[str]] = Field(default_factory=dict, description="{'matched': [...], 'missing': [...]}")
    missing_keywords: List[str] = Field(default_factory=list)
    enhanced_bullets: List[Dict[str, str]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list, description="Actionable next steps")
    executive_summary: str = Field(default="", description="High-level summary paragraph")

# LangGraph state schema
class GraphState(TypedDict, total=False):
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

# ─────────────────────────────────────────────────────────────────────────────
# 2. HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def parse_json_from_response(response_text: str) -> dict:
    """Extracts and parses JSON dictionary from LLM markdown responses."""
    json_match = re.search(r'```(?:json)?\s*\n?(\{.*?\})\s*```', response_text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))
    
    json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))
    
    raise ValueError(f"Could not extract JSON from LLM response:\n{response_text[:300]}...")

def extract_text_from_pdf_data(pdf_bytes: bytes) -> str:
    """Extracts raw text from uploaded PDF bytes using PyMuPDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = ""
    for page_num in range(len(doc)):
        page = doc[page_num]
        full_text += page.get_text("text") + "\n"
    doc.close()
    
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    return full_text.strip()

def compile_latex_to_pdf(tex_content: str, output_filename: str = "optimized_resume") -> Optional[str]:
    """Compiles LaTeX code into a PDF locally. Returns the path to PDF, or None if failed."""
    tex_path = f"{output_filename}.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)
        
    pdflatex_path = shutil.which("pdflatex")
    if not pdflatex_path:
        common_paths = [
            r"C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe",
            r"C:\Program Files (x86)\MiKTeX\miktex\bin\x64\pdflatex.exe",
            r"C:\texlive\2026\bin\windows\pdflatex.exe",
            r"C:\texlive\2025\bin\windows\pdflatex.exe",
        ]
        for p in common_paths:
            if os.path.exists(p):
                pdflatex_path = p
                break
                
    if not pdflatex_path:
        return None
        
    try:
        # Run pdflatex twice for compiling labels/references
        for _ in range(2):
            result = subprocess.run(
                [pdflatex_path, "-interaction=nonstopmode", "-halt-on-error", tex_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
        pdf_path = f"{output_filename}.pdf"
        if os.path.exists(pdf_path):
            return pdf_path
    except Exception:
        pass
    return None

# ─────────────────────────────────────────────────────────────────────────────
# 3. LANGGRAPH AGENTS
# ─────────────────────────────────────────────────────────────────────────────

def resume_parser_agent(state: GraphState) -> dict:
    resume_text = state["resume_text"]
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
- Extract ONLY information explicitly stated in the resume. Do NOT invent or hallucinate.
- List ALL skills mentioned anywhere in the resume."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Parse the following resume:\n\n{resume_text}")
    ])
    
    parsed = parse_json_from_response(response.content)
    resume_data = ResumeData(**parsed)
    return {"resume_data": resume_data.model_dump()}

def jd_parser_agent(state: GraphState) -> dict:
    jd_text = state["jd_text"]
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
- Capture the full experience requirement string."""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Parse the following job description:\n\n{jd_text}")
    ])
    
    parsed = parse_json_from_response(response.content)
    jd_data = JDData(**parsed)
    return {"jd_data": jd_data.model_dump()}

def ats_scoring_agent(state: GraphState, score_key: str = "ats_score_before") -> dict:
    resume_data = state["resume_data"]
    jd_data = state["jd_data"]
    resume_text = state["resume_text"]
    
    optimized_context = ""
    if score_key == "ats_score_after" and state.get("optimized_resume"):
        optimized = state["optimized_resume"]
        enhanced_text = "\n".join([
            f"- {b['enhanced']}" for b in optimized.get("enhanced_bullets", [])
        ])
        optimized_context = f"\n\n=== OPTIMIZED RESUME BULLETS (use these for scoring) ===\n{enhanced_text}\n\nScore the resume AS IF these enhanced bullet points replaced the originals."
    
    system_prompt = """You are an expert ATS (Applicant Tracking System) scoring engine.
Evaluate the resume against the job description strictly on these five distinct dimensions:

1. skill_match (0-100): Percentage of required technical tools, libraries, and frameworks present.
2. keyword_coverage (0-100): Alignment of specific domain phrases and methodology terms.
3. experience_alignment (0-100): Does total years of experience and seniority match the JD?
4. resume_quality (0-100): Use of action verbs, quantified metrics, and STAR/XYZ formatting.
5. project_relevance (0-100): Practical alignment of projects to JD daily tasks.

Compute the final overall_score using this industry-weighted formula:
  overall_score = (skill_match * 0.25) + (keyword_coverage * 0.15) + (experience_alignment * 0.25) + (resume_quality * 0.15) + (project_relevance * 0.20)

Return ONLY a valid JSON object matching this schema:
{
  "overall_score": <float>,
  "breakdown": {
    "skill_match": <float>,
    "keyword_coverage": <float>,
    "experience_alignment": <float>,
    "resume_quality": <float>,
    "project_relevance": <float>
  },
  "reasoning": "Explicit breakdown of the scoring."
}"""

    human_message = f"""=== RESUME DATA ===
{json.dumps(resume_data, indent=2)} 

=== RESUME FULL TEXT ===
{resume_text}

=== JOB DESCRIPTION DATA ===
{json.dumps(jd_data, indent=2)}{optimized_context}"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_message)
    ])
    
    parsed = parse_json_from_response(response.content)
    ats_score = ATSScore(**parsed)
    return {score_key: ats_score.model_dump()}

def ats_score_before_node(state: GraphState) -> dict:
    return ats_scoring_agent(state, score_key="ats_score_before")

def ats_score_after_node(state: GraphState) -> dict:
    return ats_scoring_agent(state, score_key="ats_score_after")

def gap_analysis_agent(state: GraphState) -> dict:
    resume_data = state["resume_data"]
    jd_data = state["jd_data"]
    resume_text = state["resume_text"]
    
    system_prompt = """You are an expert career advisor and resume gap analyzer.
Compare the candidate's resume against the job description and identify gaps.
Return ONLY a valid JSON object:
{
  "matched_skills": ["Skills found in BOTH"],
  "missing_skills": ["JD skills NOT found in the resume"],
  "missing_keywords": ["Important JD keywords absent from resume"],
  "weak_areas": ["Specific resume sections needing improvement, with reasons"]
}"""

    human_message = f"""=== RESUME DATA ===
{json.dumps(resume_data, indent=2)}

=== RESUME FULL TEXT ===
{resume_text}

=== JOB DESCRIPTION DATA ===
{json.dumps(jd_data, indent=2)}"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_message)
    ])
    
    parsed = parse_json_from_response(response.content)
    gap = GapAnalysis(**parsed)
    return {"gap_analysis": gap.model_dump()}

def resume_enhancement_agent(state: GraphState) -> dict:
    resume_data = state["resume_data"]
    jd_data = state["jd_data"]
    gap_analysis = state["gap_analysis"]
    
    system_prompt = """You are an expert resume writer.
Improve weak resume bullet points using STAR or XYZ format.
STAR: Situation → Task → Action → Result
XYZ: Accomplished [X] as measured by [Y] by doing [Z]

=== RULES ===
1. NEVER invent skills, experience, or fake metrics that do not exist.
2. ONLY improve wording, structure, quantification, and presentation.
3. Incorporate missing JD keywords naturally.

Return ONLY a valid JSON object:
{
  "enhanced_bullets": [
    {
      "original": "Original bullet point",
      "enhanced": "Enhanced bullet point using STAR/XYZ",
      "section": "experience | projects"
    }
  ],
  "keywords_incorporated": ["JD keywords woven in"],
  "optimization_summary": "Brief summary of improvements"
}"""

    human_message = f"""=== RESUME DATA ===
{json.dumps(resume_data, indent=2)}

=== JOB DESCRIPTION DATA ===
{json.dumps(jd_data, indent=2)}

=== GAP ANALYSIS ===
{json.dumps(gap_analysis, indent=2)}"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_message)
    ])
    
    parsed = parse_json_from_response(response.content)
    optimized = OptimizedResume(**parsed)
    return {"optimized_resume": optimized.model_dump()}

def report_generator_agent(state: GraphState) -> dict:
    ats_before = state["ats_score_before"]
    ats_after = state["ats_score_after"]
    gap = state["gap_analysis"]
    optimized = state["optimized_resume"]
    
    score_before = ats_before["overall_score"]
    score_after = ats_after["overall_score"]
    improvement = ((score_after - score_before) / max(score_before, 1)) * 100
    
    system_prompt = """You are an expert career advisor generating a final resume optimization report.
Return ONLY a valid JSON object matching this schema:
{
  "ats_score_before": <float>,
  "ats_score_after": <float>,
  "improvement_percentage": <float>,
  "skill_match_analysis": {
    "matched": ["skill1", "skill2"],
    "missing": ["skill3", "skill4"]
  },
  "missing_keywords": ["kw1", "kw2"],
  "enhanced_bullets": [
    {"original": "...", "enhanced": "...", "section": "..."}
  ],
  "recommendations": [
    "actionable recommendation 1",
    "actionable recommendation 2"
  ],
  "executive_summary": "A 2-3 sentence high-level summary of the findings."
}"""

    human_message = f"""=== ATS SCORE BEFORE ===
{json.dumps(ats_before, indent=2)}

=== ATS SCORE AFTER ===
{json.dumps(ats_after, indent=2)}

=== GAP ANALYSIS ===
{json.dumps(gap, indent=2)}

=== OPTIMIZED BULLETS ===
{json.dumps(optimized, indent=2)}"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_message)
    ])
    
    parsed = parse_json_from_response(response.content)
    report = FinalReport(**parsed)
    return {"final_report": report.model_dump()}

def resume_builder_agent(state: GraphState) -> dict:
    optimized_resume = state["optimized_resume"]
    gap_analysis = state["gap_analysis"]
    
    template_path = "resume laytex format.tex"
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            latex_template = f.read()
    else:
        # Fallback template
        latex_template = "% Base Template 'resume laytex format.tex' Not Found"
        
    system_prompt = """You are an expert LaTeX typographer. Take the provided LaTeX template and update it with the optimized resume content.

=== RULES ===
1. Keep the exact document class, packages, geometry, styling, fonts, and custom commands from the template.
2. In the Projects, Experience, and Education sections, replace original bullet points with the enhanced bullet points from the Optimized Resume data. Keep un-enhanced bullets as-is.
3. Weave the missing skills/keywords that were successfully incorporated into the Technical Skills section. Do NOT remove existing skills.
4. Ensure all special LaTeX characters (such as %, &) in the new text are properly escaped (e.g. write '%' as '\\%' and '&' as '\\&').
5. Return ONLY the raw LaTeX code. Do NOT wrap it in markdown code fences."""

    human_message = f"""=== BASE LATEX TEMPLATE ===
{latex_template}

=== OPTIMIZED BULLET POINTS ===
{json.dumps(optimized_resume.get('enhanced_bullets', []), indent=2)}

=== GAP ANALYSIS (INCORPORATE KEYWORDS/SKILLS) ===
{json.dumps(gap_analysis, indent=2)}"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_message)
    ])
    
    tex_content = response.content.strip()
    if tex_content.startswith("```"):
        lines = tex_content.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines[-1].startswith("```"):
            lines = lines[:-1]
        tex_content = "\n".join(lines).strip()
        
    # Compile the LaTeX to PDF
    pdf_path = compile_latex_to_pdf(tex_content, "optimized_resume")
    
    return {"optimized_latex": tex_content}

# ─────────────────────────────────────────────────────────────────────────────
# 4. BUILD LANGGRAPH WORKFLOW
# ─────────────────────────────────────────────────────────────────────────────

def build_workflow():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("resume_parser", resume_parser_agent)
    workflow.add_node("jd_parser", jd_parser_agent)
    workflow.add_node("ats_score_before", ats_score_before_node)
    workflow.add_node("gap_analysis", gap_analysis_agent)
    workflow.add_node("resume_enhancement", resume_enhancement_agent)
    workflow.add_node("ats_score_after", ats_score_after_node)
    workflow.add_node("report_generator", report_generator_agent)
    workflow.add_node("resume_builder", resume_builder_agent)
    
    workflow.add_edge(START, "resume_parser")
    workflow.add_edge("resume_parser", "jd_parser")
    workflow.add_edge("jd_parser", "ats_score_before")
    workflow.add_edge("ats_score_before", "gap_analysis")
    workflow.add_edge("gap_analysis", "resume_enhancement")
    workflow.add_edge("resume_enhancement", "ats_score_after")
    workflow.add_edge("ats_score_after", "report_generator")
    workflow.add_edge("report_generator", "resume_builder")
    workflow.add_edge("resume_builder", END)
    
    return workflow.compile()

# ─────────────────────────────────────────────────────────────────────────────
# 5. STREAMLIT APP LAYOUT & LOGIC
# ─────────────────────────────────────────────────────────────────────────────

# Configure Groq LLM globally from environment variables / .env file
groq_api_key = os.getenv("GROQ_API_KEY", "")
if groq_api_key:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_api_key,
        temperature=0.2,
        max_tokens=8192
    )
else:
    llm = None

# App Header
st.title("🚀 AI-Powered Resume-to-JD Gap Analyzer & Builder")
st.markdown("Optimize your resume, close the skill gaps, and build an ATS-friendly LaTeX resume in seconds.")
st.markdown("---")

# Input Section (Two-Column Layout)
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📤 Upload Resume")
    uploaded_file = st.file_uploader(
        "Upload your current resume (PDF format only)", 
        type=["pdf"],
        help="Drag and drop or select a PDF file."
    )
    
    # Custom File Type Error Handling (in case they bypass type constraint)
    if uploaded_file is not None:
        if not uploaded_file.name.lower().endswith(".pdf"):
            st.error("❌ Unsupported file format. Please upload a PDF resume.")
            uploaded_file = None

with col_right:
    st.subheader("📋 Paste Job Description")
    jd_input = st.text_area(
        "Paste the Job Description (JD) text here",
        height=180,
        placeholder="Enter the job description details..."
    )

# Execution Steps Configuration for Progress Display
NODE_LOGS = {
    "resume_parser": "Resume parsed into structured sections ✅",
    "jd_parser": "Job description parsed & requirements extracted ✅",
    "ats_score_before": "Initial ATS score calculated ✅",
    "gap_analysis": "Skill gap analysis completed ✅",
    "resume_enhancement": "Resume enhancement started ✅",
    "ats_score_after": "Post-optimization ATS score calculated ✅",
    "report_generator": "Comparison report generated ✅",
    "resume_builder": "LaTeX resume compiled & saved ✅"
}

# Run Analysis Button
if st.button("🚀 Analyze & Optimize Resume", use_container_width=True):
    if not os.environ.get("GROQ_API_KEY"):
        st.error("❌ Groq API key is missing. Please set the GROQ_API_KEY environment variable or define it in your .env file.")
    elif uploaded_file is None:
        st.error("❌ Please upload a valid PDF resume first.")
    elif not jd_input.strip():
        st.error("❌ Please paste the job description first.")
    else:
        try:
            # 1. Start thinking logs
            st.markdown("<div class='section-title'>🧠 AI Agent Execution Flow</div>", unsafe_allow_html=True)
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Status logger container
            log_container = st.container()
            with log_container:
                st.info("📤 Resume file uploaded successfully.")
                time.sleep(0.3)
                st.info("📄 Extracting text from PDF resume...")
                
                # PDF Text Extraction
                pdf_bytes = uploaded_file.read()
                resume_text = extract_text_from_pdf_data(pdf_bytes)
                st.info(f"📄 Resume text extracted: {len(resume_text)} characters.")
                time.sleep(0.3)
                
            # Compile graph
            app = build_workflow()
            
            # Initial LangGraph state
            initial_state = {
                "resume_text": resume_text,
                "jd_text": jd_input
            }
            
            # Execution streaming with progress bar updates
            progress_mapping = {
                "resume_parser": 15,
                "jd_parser": 30,
                "ats_score_before": 45,
                "gap_analysis": 60,
                "resume_enhancement": 75,
                "ats_score_after": 85,
                "report_generator": 95,
                "resume_builder": 100
            }
            
            # Run pipeline
            final_state = {}
            for chunk in app.stream(initial_state):
                for node_name, state_update in chunk.items():
                    # Accumulate state updates
                    final_state.update(state_update)
                    
                    # Update status logs dynamically
                    if node_name in NODE_LOGS:
                        log_container.success(NODE_LOGS[node_name])
                        progress_val = progress_mapping.get(node_name, 100)
                        progress_bar.progress(progress_val)
                        status_text.markdown(f"**Step progress: {progress_val}%**")
                        time.sleep(0.2)
            
            # Complete execution indicator
            status_text.markdown("**Step progress: 100% — Analysis Complete! 🎉**")
            st.success("✅ Optimization pipeline completed successfully!")
            
            # ─────────────────────────────────────────────────────────────────
            # 6. RESULTS DASHBOARD
            # ─────────────────────────────────────────────────────────────────
            st.markdown("<div class='section-title'>📊 Resume Analysis Dashboard</div>", unsafe_allow_html=True)
            
            report = final_state.get("final_report", {})
            score_before = report.get("ats_score_before", 0)
            score_after = report.get("ats_score_after", 0)
            improvement = report.get("improvement_percentage", 0)
            
            # Metric Columns
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Initial ATS Score", f"{score_before:.1f}/100")
            with m_col2:
                st.metric("Optimized ATS Score", f"{score_after:.1f}/100")
            with m_col3:
                st.metric("ATS Match Improvement", f"{improvement:+.1f}%")
                
            # Content Tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Score Details", 
                "🔍 Skill Matches & Gaps", 
                "✨ Bullet Point Improvements", 
                "📝 Actionable Recommendations",
                "💾 Download Center"
            ])
            
            with tab1:
                st.markdown("### Category-wise ATS Score Comparison")
                before_breakdown = final_state.get("ats_score_before", {}).get("breakdown", {})
                after_breakdown = final_state.get("ats_score_after", {}).get("breakdown", {})
                
                # Show scores as a table
                score_data = {
                    "Metric": [
                        "🔧 Skill Match", 
                        "🔑 Keyword Coverage", 
                        "👔 Experience Alignment", 
                        "✍️ Resume Wording Quality", 
                        "📁 Project Relevance"
                    ],
                    "Before Score": [
                        before_breakdown.get("skill_match", 0),
                        before_breakdown.get("keyword_coverage", 0),
                        before_breakdown.get("experience_alignment", 0),
                        before_breakdown.get("resume_quality", 0),
                        before_breakdown.get("project_relevance", 0),
                    ],
                    "After Score": [
                        after_breakdown.get("skill_match", 0),
                        after_breakdown.get("keyword_coverage", 0),
                        after_breakdown.get("experience_alignment", 0),
                        after_breakdown.get("resume_quality", 0),
                        after_breakdown.get("project_relevance", 0),
                    ]
                }
                st.table(score_data)
                
            with tab2:
                # Skill Analysis Columns
                skill_cols = report.get("skill_match_analysis", {})
                matched_skills = skill_cols.get("matched", [])
                missing_skills = skill_cols.get("missing", [])
                missing_keywords = report.get("missing_keywords", [])
                
                sk_col1, sk_col2 = st.columns(2)
                with sk_col1:
                    st.markdown("#### ✅ Matched Skills")
                    if matched_skills:
                        for sk in matched_skills:
                            st.write(f"- {sk}")
                    else:
                        st.write("No direct matching skills found.")
                        
                with sk_col2:
                    st.markdown("#### ❌ Missing Skills")
                    if missing_skills:
                        for sk in missing_skills:
                            st.markdown(f"- <span style='color:#ff7676'>{sk}</span>", unsafe_allow_html=True)
                    else:
                        st.write("No missing skills identified! Great job!")
                        
                st.markdown("#### 🔑 Missing JD Keywords")
                if missing_keywords:
                    st.write(", ".join([f"**{kw}**" for kw in missing_keywords]))
                else:
                    st.write("No major keywords missing.")
                    
            with tab3:
                st.markdown("### Side-by-Side Bullet Point Enhancements")
                st.caption("Here is how the AI improved your weaker bullet points using STAR or XYZ format to improve ATS matching:")
                
                enhanced_bullets = report.get("enhanced_bullets", [])
                if enhanced_bullets:
                    # Render custom HTML table
                    table_html = "<table class='comparison-table'><thead><tr><th>Section</th><th>Original Bullet Point</th><th>ATS-Optimized Bullet Point</th></tr></thead><tbody>"
                    for b in enhanced_bullets:
                        table_html += f"""
                        <tr>
                            <td><strong>{b.get('section', 'experience').capitalize()}</strong></td>
                            <td class='original-bullet'>{b.get('original', '')}</td>
                            <td class='enhanced-bullet'>{b.get('enhanced', '')}</td>
                        </tr>
                        """
                    table_html += "</tbody></table>"
                    st.markdown(table_html, unsafe_allow_html=True)
                else:
                    st.write("No bullet points needed major enhancement.")
                    
            with tab4:
                st.markdown("### Actionable Next Steps")
                
                # Executive summary
                st.markdown("#### Summary Analysis")
                st.write(report.get("executive_summary", "Summary not available."))
                
                # Recommendations list
                st.markdown("#### Recommendations")
                recommendations = report.get("recommendations", [])
                if recommendations:
                    for idx, rec in enumerate(recommendations, 1):
                        st.write(f"**{idx}.** {rec}")
                else:
                    st.write("No recommendations provided.")
                    
            with tab5:
                st.markdown("### Download Your Optimized Resume")
                
                tex_content = final_state.get("optimized_latex", "")
                
                if tex_content:
                    d_col1, d_col2 = st.columns(2)
                    with d_col1:
                        st.download_button(
                            label="📄 Download LaTeX Code (.tex)",
                            data=tex_content,
                            file_name="optimized_resume.tex",
                            mime="text/plain",
                            use_container_width=True
                        )
                        st.info("💡 You can open the downloaded `.tex` file in [Overleaf](https://www.overleaf.com) to render and download your final PDF resume.")
                        
                    with d_col2:
                        pdf_path = "optimized_resume.pdf"
                        if os.path.exists(pdf_path):
                            with open(pdf_path, "rb") as pdf_file:
                                pdf_data = pdf_file.read()
                            st.download_button(
                                label="🎓 Download Compiled PDF Resume (.pdf)",
                                data=pdf_data,
                                file_name="optimized_resume.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                            st.success("✅ Compiled PDF is ready for download!")
                        else:
                            st.warning("⚠️ Local PDF compilation was not successful or pdflatex was not found. Please download the LaTeX source above and compile it using Overleaf.")
                            st.button(
                                label="🎓 Download Compiled PDF Resume (.pdf)",
                                disabled=True,
                                use_container_width=True
                            )
                else:
                    st.error("LaTeX resume content could not be generated. Please check the logs.")
                    
        except Exception as e:
            st.error("❌ An error occurred during pipeline execution.")
            st.error(f"Error details: {e}")
            st.info("💡 Please verify that your Groq API key is correct and you have not exceeded your API limits. Try running again.")
