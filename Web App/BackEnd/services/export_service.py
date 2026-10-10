import json
import uuid
import re
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from config import TEMP_DIR, LATEX_TEMPLATE_PATH
from core.llm import get_llm_with_fallback
from utils.latex_compiler import compile_latex_to_pdf, is_latex_compiler_available

logger = logging.getLogger("ai_resume_optimizer.export_service")


# ─── LaTeX Formatting Utilities ──────────────────────────────────────────────
def escape_latex(text: str) -> str:
    """Escapes special LaTeX characters while preventing double-escaping."""
    if not text:
        return ""
    
    # Temporarily preserve existing escape sequences
    text = text.replace(r'\%', '__ESCPERCENT__')
    text = text.replace(r'\&', '__ESCAMP__')
    text = text.replace(r'\_', '__ESCUND__')
    text = text.replace(r'\$', '__ESCDOLLAR__')
    text = text.replace(r'\#', '__ESCHASH__')

    # Escape raw special characters
    text = text.replace('%', r'\%')
    text = text.replace('&', r'\&')
    text = text.replace('_', r'\_')
    text = text.replace('$', r'\$')
    text = text.replace('#', r'\#')

    # Restore preserved escapes
    text = text.replace('__ESCPERCENT__', r'\%')
    text = text.replace('__ESCAMP__', r'\&')
    text = text.replace('__ESCUND__', r'\_')
    text = text.replace('__ESCDOLLAR__', r'\$')
    text = text.replace('__ESCHASH__', r'\#')
    return text


def strip_think_tags(text: str) -> str:
    """Strips <think>...</think> blocks from reasoning model outputs."""
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    elif "<think>" in text:
        text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    return text


# ─── DOCX XML Styling Helpers ────────────────────────────────────────────────
def add_bottom_border(paragraph):
    """Adds a sleek bottom border line under a paragraph like a LaTeX titlerule."""
    pPr = paragraph._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), '6')
    bottom.set(qn('w:space'), '2')
    bottom.set(qn('w:color'), '747878')
    pBdr.append(bottom)
    pPr.append(pBdr)


def add_two_column_line(doc, left_text, right_text, left_bold=True, right_italic=True, left_size=10.5, right_size=10):
    """Creates an aligned line with text on the left and text on the right."""
    table = doc.add_table(rows=1, cols=2)
    table.autofit = False
    table.columns[0].width = Inches(5.3)
    table.columns[1].width = Inches(2.0)
    
    cell_left = table.cell(0, 0)
    cell_right = table.cell(0, 1)
    
    p_left = cell_left.paragraphs[0]
    p_left.paragraph_format.space_before = Pt(2)
    p_left.paragraph_format.space_after = Pt(1)
    run_l = p_left.add_run(left_text)
    run_l.bold = left_bold
    run_l.font.size = Pt(left_size)
    run_l.font.name = 'Calibri'
    
    p_right = cell_right.paragraphs[0]
    p_right.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_right.paragraph_format.space_before = Pt(2)
    p_right.paragraph_format.space_after = Pt(1)
    run_r = p_right.add_run(right_text)
    run_r.italic = right_italic
    run_r.font.size = Pt(right_size)
    run_r.font.name = 'Calibri'
    
    # Strip borders and padding
    for row in table.rows:
        trPr = row._tr.get_or_add_trPr()
        trPr.append(OxmlElement('w:cantSplit'))
        for cell in row.cells:
            tcPr = cell._tc.get_or_add_tcPr()
            tcMar = OxmlElement('w:tcMar')
            for m in ['top', 'bottom', 'left', 'right']:
                node = OxmlElement(f'w:{m}')
                node.set(qn('w:w'), '0')
                node.set(qn('w:type'), 'dxa')
                tcMar.append(node)
            tcPr.append(tcMar)


def create_session_dir(session_id: Optional[str] = None) -> Tuple[str, Path]:
    """Generates an isolated directory for a user export session."""
    sid = session_id or uuid.uuid4().hex[:12]
    session_dir = TEMP_DIR / sid
    session_dir.mkdir(parents=True, exist_ok=True)
    return sid, session_dir


# ─── Robust LaTeX Synthesis ──────────────────────────────────────────────────
def synthesize_latex(
    template_str: str,
    optimized_resume: Dict[str, Any],
    summary_text: Optional[str] = None
) -> str:
    """
    Surgically injects enhanced bullets, incorporated keywords, and summary
    into the ATS LaTeX template while preserving all macros and formatting.
    """
    tex = template_str
    enhanced_bullets = optimized_resume.get("enhanced_bullets", [])
    keywords_to_add = optimized_resume.get("keywords_incorporated", [])

    # 1. Update Professional Summary if provided
    if summary_text and summary_text.strip():
        # Match paragraph after \section{Professional Summary}
        sum_pattern = r"(\\section\{Professional Summary\}\s*\n+)([\s\S]*?)(\n+\\vspace|\n+\\section)"
        escaped_summary = escape_latex(summary_text.strip())
        if re.search(sum_pattern, tex):
            tex = re.sub(sum_pattern, rf"\g<1>{escaped_summary}\n\g<3>", tex, count=1)

    # 2. Weave missing keywords into Technical Skills
    if keywords_to_add:
        # Categorize keywords
        deploy_keywords = []
        ml_keywords = []
        data_keywords = []
        lang_keywords = []

        for kw in keywords_to_add:
            kw_clean = kw.strip()
            kw_lower = kw_clean.lower()
            if any(term in kw_lower for term in ["docker", "kubernetes", "ci/cd", "k8s", "fastapi", "flask", "aws", "gcp", "azure", "mlflow", "vertex", "sagemaker"]):
                deploy_keywords.append(kw_clean)
            elif any(term in kw_lower for term in ["pytorch", "transformers", "bert", "llm", "langchain", "langgraph", "nlp", "vision", "keras", "tensorflow"]):
                ml_keywords.append(kw_clean)
            elif any(term in kw_lower for term in ["sql", "python", "r", "c++", "scala", "java"]):
                lang_keywords.append(kw_clean)
            else:
                data_keywords.append(kw_clean)

        # Append to matching lines in LaTeX template
        if deploy_keywords and r"\textbf{Deployment \& Version Control:}" in tex:
            addition = ", ".join(escape_latex(k) for k in deploy_keywords)
            tex = tex.replace(
                r"\textbf{Deployment \& Version Control:} Streamlit, Git / GitHub",
                rf"\textbf{{Deployment \& Version Control:}} Streamlit, Git / GitHub, {addition}"
            )
        if ml_keywords and r"\textbf{ML / DL Frameworks:}" in tex:
            addition = ", ".join(escape_latex(k) for k in ml_keywords)
            tex = tex.replace(
                r"\textbf{ML / DL Frameworks:} Scikit-learn, TensorFlow, Keras, XGBoost, Random Forest, Gradient Boosting",
                rf"\textbf{{ML / DL Frameworks:}} Scikit-learn, TensorFlow, Keras, XGBoost, Random Forest, Gradient Boosting, {addition}"
            )
        if data_keywords and r"\textbf{Data Tools:}" in tex:
            addition = ", ".join(escape_latex(k) for k in data_keywords)
            tex = tex.replace(
                r"\textbf{Data Tools:} Pandas, NumPy, Matplotlib, Seaborn, SHAP, SMOTE, Jupyter",
                rf"\textbf{{Data Tools:}} Pandas, NumPy, Matplotlib, Seaborn, SHAP, SMOTE, Jupyter, {addition}"
            )

    # 3. Replace matching bullet points with enhanced versions
    for b in enhanced_bullets:
        orig = b.get("original", "").strip()
        enh = b.get("enhanced", "").strip()
        if not orig or not enh:
            continue

        # Extract anchor keywords (first 25 characters of the original bullet)
        anchor = re.sub(r'[^a-zA-Z0-9\s]', '', orig[:30]).strip()
        if not anchor:
            continue

        lines = tex.splitlines()
        replaced = False
        for i, line in enumerate(lines):
            line_clean = re.sub(r'[^a-zA-Z0-9\s]', '', line)
            if anchor.lower() in line_clean.lower() and r"\item" in line:
                lines[i] = f"  \\item {escape_latex(enh)}"
                replaced = True
                break

        if replaced:
            tex = "\n".join(lines)

    return tex


def generate_latex_source(
    resume_data: Dict[str, Any],
    optimized_resume: Dict[str, Any],
    api_key: Optional[str] = None
) -> str:
    """
    Synthesizes the optimized resume into clean LaTeX code using the base template.
    Guarantees non-empty output with automatic fallback.
    """
    template_path = LATEX_TEMPLATE_PATH
    latex_template = ""
    if template_path.exists():
        with open(template_path, "r", encoding="utf-8") as f:
            latex_template = f.read()

    if not latex_template.strip():
        logger.error(f"Template at {template_path} is empty or missing.")
        raise FileNotFoundError(f"LaTeX template missing at: {template_path}")

    # Programmatically weave bullets & keywords into template
    synthesized_tex = synthesize_latex(
        template_str=latex_template,
        optimized_resume=optimized_resume,
        summary_text=optimized_resume.get("optimization_summary")
    )

    # Verify synthesized LaTeX
    if r"\documentclass" in synthesized_tex and r"\end{document}" in synthesized_tex:
        logger.info(f"Programmatic LaTeX synthesis succeeded ({len(synthesized_tex)} bytes).")
        return synthesized_tex

    return latex_template


# ─── Company-Ready ATS DOCX Generator ─────────────────────────────────────────
def generate_docx_document(
    resume_data: Dict[str, Any],
    optimized_resume: Dict[str, Any],
    output_path: Path
) -> Path:
    """
    Generates a company-ready, beautifully styled single-column ATS resume in Word format
    matching the exact visual layout of 'resume laytex format.tex'.
    """
    doc = Document()
    
    # 1. Page Margins (0.5 in top/bottom, 0.6 in left/right)
    for s in doc.sections:
        s.top_margin = Inches(0.5)
        s.bottom_margin = Inches(0.5)
        s.left_margin = Inches(0.6)
        s.right_margin = Inches(0.6)

    # 2. Header (Name & Contact)
    p_name = doc.add_paragraph()
    p_name.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_name.paragraph_format.space_before = Pt(0)
    p_name.paragraph_format.space_after = Pt(2)
    r_name = p_name.add_run("HIMAKESH KRISHNA")
    r_name.bold = True
    r_name.font.size = Pt(18)
    r_name.font.name = 'Calibri'

    p_contact = doc.add_paragraph()
    p_contact.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_contact.paragraph_format.space_before = Pt(0)
    p_contact.paragraph_format.space_after = Pt(6)
    r_contact = p_contact.add_run("+91 7979777206  |  dr.himakesh@gmail.com  |  linkedin.com/in/himakesh-krishna  |  github.com/davidkumar355")
    r_contact.font.size = Pt(9.5)
    r_contact.font.color.rgb = RGBColor(0x44, 0x47, 0x48)
    r_contact.font.name = 'Calibri'

    # 3. Section: Professional Summary
    p_sec1 = doc.add_paragraph()
    p_sec1.paragraph_format.space_before = Pt(6)
    p_sec1.paragraph_format.space_after = Pt(4)
    r_sec1 = p_sec1.add_run("PROFESSIONAL SUMMARY")
    r_sec1.bold = True
    r_sec1.font.size = Pt(11)
    r_sec1.font.name = 'Calibri'
    add_bottom_border(p_sec1)

    p_sum = doc.add_paragraph()
    p_sum.paragraph_format.space_before = Pt(2)
    p_sum.paragraph_format.space_after = Pt(6)
    p_sum.paragraph_format.line_spacing = 1.15
    summary_text = (
        "Data Science practitioner (IIT Guwahati PG Certification, 2025–2026) with hands-on experience building "
        "end-to-end ML pipelines across NLP, computer vision, and unsupervised learning — including a 95% accuracy BBC News classifier, "
        "a 95.52% CIFAR-10 model with Grad-CAM explainability, and a customer segmentation dashboard deployed on Streamlit. "
        "Six years of professional motion graphics and graphic design experience informs a rare strength in data visualization "
        "and communicating complex insights to non-technical stakeholders."
    )
    r_sum = p_sum.add_run(summary_text)
    r_sum.font.size = Pt(10)
    r_sum.font.name = 'Calibri'

    # 4. Section: Technical Skills
    p_sec2 = doc.add_paragraph()
    p_sec2.paragraph_format.space_before = Pt(6)
    p_sec2.paragraph_format.space_after = Pt(4)
    r_sec2 = p_sec2.add_run("TECHNICAL SKILLS")
    r_sec2.bold = True
    r_sec2.font.size = Pt(11)
    r_sec2.font.name = 'Calibri'
    add_bottom_border(p_sec2)

    # Added keywords
    added_keywords = optimized_resume.get("keywords_incorporated", [])
    added_str = f", {', '.join(added_keywords)}" if added_keywords else ""

    skills_categories = [
        ("Languages:", "Python, SQL (intermediate)"),
        ("ML / DL Frameworks:", "Scikit-learn, TensorFlow, Keras, XGBoost, Random Forest, Gradient Boosting"),
        ("NLP:", "NLTK, Word2Vec, HuggingFace (T5, BERT), BiLSTM, Seq2Seq, Bahdanau Attention"),
        ("Computer Vision:", "CNN, EfficientNetV2M, DenseNet121, Transfer Learning, Grad-CAM"),
        ("Unsupervised ML:", "K-Means, GMM, Agglomerative Clustering, DBSCAN, PCA"),
        ("Data Tools:", "Pandas, NumPy, Matplotlib, Seaborn, SHAP, SMOTE, Jupyter"),
        ("Deployment & Version Control:", f"Streamlit, Git / GitHub{added_str}"),
        ("Other:", "Feature Engineering, EDA, Statistical Analysis, Class Imbalance Handling"),
    ]

    for label, val in skills_categories:
        p_sk = doc.add_paragraph()
        p_sk.paragraph_format.space_before = Pt(1)
        p_sk.paragraph_format.space_after = Pt(1)
        r_lbl = p_sk.add_run(label + " ")
        r_lbl.bold = True
        r_lbl.font.size = Pt(9.5)
        r_lbl.font.name = 'Calibri'
        r_val = p_sk.add_run(val)
        r_val.font.size = Pt(9.5)
        r_val.font.name = 'Calibri'

    # 5. Section: Projects
    p_sec3 = doc.add_paragraph()
    p_sec3.paragraph_format.space_before = Pt(8)
    p_sec3.paragraph_format.space_after = Pt(4)
    r_sec3 = p_sec3.add_run("PROJECTS")
    r_sec3.bold = True
    r_sec3.font.size = Pt(11)
    r_sec3.font.name = 'Calibri'
    add_bottom_border(p_sec3)

    # Map of enhanced bullets for quick lookup
    enhanced_map = {
        b.get("original", "").strip().lower()[:30]: b.get("enhanced", "").strip()
        for b in optimized_resume.get("enhanced_bullets", [])
        if b.get("original") and b.get("enhanced")
    }

    projects_data = [
        {
            "name": "BBC News Multi-Class NLP Classification & Summarization System",
            "year": "2025",
            "stack": "Python, Logistic Regression, BiLSTM, Word2Vec, T5-small, HuggingFace, Keras, Streamlit",
            "bullets": [
                "Built an end-to-end NLP pipeline on 2,225 BBC News articles; Logistic Regression achieved 95% accuracy and Macro F1 of 0.87 across 5 categories (Tech, Politics, Business, Sports, Entertainment).",
                "Resolved severe class imbalance via class-weighting strategy, improving minority Technology-class Recall from 12% to 64% (433% improvement) in the BiLSTM model.",
                "Fine-tuned T5-small for abstractive headline generation and implemented Bahdanau attention-based Seq2Seq model — spanning classical-to-transformer NLP in a single pipeline.",
                "Deployed interactive Streamlit inference dashboard enabling real-time article classification and headline generation."
            ]
        },
        {
            "name": "CIFAR-10 Computer Vision Pipeline with Explainability",
            "year": "2025",
            "stack": "TensorFlow, Keras, EfficientNetV2M, DenseNet121, Custom CNN, Grad-CAM, Streamlit",
            "bullets": [
                "Benchmarked 3 CNN architectures (Custom CNN, DenseNet121, EfficientNetV2M) on CIFAR-10 (60K images, 10 classes); identified and fixed a double-preprocessing normalization bug that suppressed accuracy by ~10%.",
                "Trained EfficientNetV2M with two-phase fine-tuning (feature extraction + gradual unfreeze), achieving 95.52% test accuracy — 6% above custom CNN baseline.",
                "Implemented Grad-CAM explainability for nested Functional API model, enabling visual inspection of model attention across all 10 classes.",
                "Deployed as interactive Streamlit app with real-time prediction and saliency-map visualization for non-technical stakeholders."
            ]
        },
        {
            "name": "Credit Card Customer Behavioral Segmentation",
            "year": "2025",
            "stack": "Python, K-Means, GMM, Agglomerative, DBSCAN, PCA, Scikit-learn, Pandas, Streamlit",
            "bullets": [
                "Engineered a customer segmentation pipeline on 8,950 credit card customers; applied winsorization, log-transform, and StandardScaler normalization to prepare 17 raw features for distance-based clustering.",
                "Designed 4 domain-specific behavioral indicators (Spend Intensity, Payment Discipline, Revolving Behavior, Cash Dependency) from raw transaction data, transforming low-signal financial fields into interpretable features.",
                "Applied PCA to retain ~96% variance while reducing dimensionality; benchmarked K-Means, GMM, Agglomerative, and DBSCAN via silhouette score, inertia, and information criteria.",
                "Identified 4 actionable customer personas (Power Spender, Disciplined Payer, Silent Revolver, Cash Dependent); deployed as 3-page Streamlit dashboard for stakeholder-ready business insights."
            ]
        },
        {
            "name": "Bank Marketing Subscription Classifier (Ensemble Methods Case Study)",
            "year": "2025",
            "stack": "Python, Scikit-learn, Random Forest, Gradient Boosting, Stacking, SMOTE, SHAP, Pandas",
            "bullets": [
                "Built a term-deposit subscription classifier on 11,162 customer records; resolved class imbalance with SMOTE and benchmarked Random Forest, Gradient Boosting, and Stacking ensembles.",
                "Stacking Ensemble achieved ROC-AUC 0.925; Random Forest achieved best F1-score of 0.853 across 5-fold stratified cross-validation.",
                "Applied SHAP explainability to identify top predictors (call duration, prior campaign outcome, contact method) and translated model outputs into targeted marketing recommendations."
            ]
        }
    ]

    for proj in projects_data:
        add_two_column_line(doc, proj["name"], proj["year"], left_bold=True, right_italic=True)
        
        p_st = doc.add_paragraph()
        p_st.paragraph_format.space_before = Pt(0)
        p_st.paragraph_format.space_after = Pt(2)
        r_st = p_st.add_run(f"Stack: {proj['stack']}")
        r_st.italic = True
        r_st.font.size = Pt(9)
        r_st.font.color.rgb = RGBColor(0x44, 0x47, 0x48)
        r_st.font.name = 'Calibri'

        for bullet in proj["bullets"]:
            bullet_key = bullet.strip().lower()[:30]
            final_text = enhanced_map.get(bullet_key, bullet)

            p_b = doc.add_paragraph(style='List Bullet')
            p_b.paragraph_format.space_before = Pt(1)
            p_b.paragraph_format.space_after = Pt(2)
            p_b.paragraph_format.line_spacing = 1.15
            r_b = p_b.add_run(final_text)
            r_b.font.size = Pt(9.5)
            r_b.font.name = 'Calibri'

    # 6. Section: Education
    p_sec4 = doc.add_paragraph()
    p_sec4.paragraph_format.space_before = Pt(8)
    p_sec4.paragraph_format.space_after = Pt(4)
    r_sec4 = p_sec4.add_run("EDUCATION")
    r_sec4.bold = True
    r_sec4.font.size = Pt(11)
    r_sec4.font.name = 'Calibri'
    add_bottom_border(p_sec4)

    add_two_column_line(doc, "Indian Institute of Technology (IIT), Guwahati", "Aug 2025 – Jun 2026")
    add_two_column_line(doc, "Post-Graduate Certification in Data Science", "Guwahati, Assam", left_bold=False, right_italic=True, left_size=9.5, right_size=9.5)
    
    p_ed1 = doc.add_paragraph(style='List Bullet')
    p_ed1.paragraph_format.space_before = Pt(1)
    p_ed1.paragraph_format.space_after = Pt(1)
    r_ed1 = p_ed1.add_run("Relevant coursework: Machine Learning, Deep Learning, NLP, Statistics & EDA, Predictive Modeling, Data Visualization")
    r_ed1.font.size = Pt(9.5)
    r_ed1.font.name = 'Calibri'

    p_ed2 = doc.add_paragraph(style='List Bullet')
    p_ed2.paragraph_format.space_before = Pt(1)
    p_ed2.paragraph_format.space_after = Pt(3)
    r_ed2 = p_ed2.add_run("Certifications earned: IITG Statistics & EDA, IITG Introduction & Programming in Data Science")
    r_ed2.font.size = Pt(9.5)
    r_ed2.font.name = 'Calibri'

    add_two_column_line(doc, "Munger University", "May 2022 – Sep 2025")
    add_two_column_line(doc, "Bachelor of Science (B.Sc.) — Zoology / Animal Biology", "Munger, Bihar", left_bold=False, right_italic=True, left_size=9.5, right_size=9.5)

    # 7. Section: Work Experience
    p_sec5 = doc.add_paragraph()
    p_sec5.paragraph_format.space_before = Pt(8)
    p_sec5.paragraph_format.space_after = Pt(4)
    r_sec5 = p_sec5.add_run("WORK EXPERIENCE")
    r_sec5.bold = True
    r_sec5.font.size = Pt(11)
    r_sec5.font.name = 'Calibri'
    add_bottom_border(p_sec5)

    add_two_column_line(doc, "Nerul Marketing", "Dec 2024 – Dec 2025")
    add_two_column_line(doc, "Graphic Designer & Motion Graphics Specialist", "Remote", left_bold=False, right_italic=True, left_size=9.5, right_size=9.5)

    p_we1 = doc.add_paragraph(style='List Bullet')
    p_we1.paragraph_format.space_before = Pt(1)
    p_we1.paragraph_format.space_after = Pt(1)
    r_we1 = p_we1.add_run("Designed 50+ data-driven motion graphic presentations for enterprise clients, translating complex performance metrics and campaign data into visual narratives for non-technical stakeholders.")
    r_we1.font.size = Pt(9.5)
    r_we1.font.name = 'Calibri'

    p_we2 = doc.add_paragraph(style='List Bullet')
    p_we2.paragraph_format.space_before = Pt(1)
    p_we2.paragraph_format.space_after = Pt(3)
    r_we2 = p_we2.add_run("Developed full creative pipelines using Photoshop, After Effects, CapCut, and AI tools — managed client brief to delivery end-to-end.")
    r_we2.font.size = Pt(9.5)
    r_we2.font.name = 'Calibri'

    # 8. Section: Certifications
    p_sec6 = doc.add_paragraph()
    p_sec6.paragraph_format.space_before = Pt(8)
    p_sec6.paragraph_format.space_after = Pt(4)
    r_sec6 = p_sec6.add_run("CERTIFICATIONS")
    r_sec6.bold = True
    r_sec6.font.size = Pt(11)
    r_sec6.font.name = 'Calibri'
    add_bottom_border(p_sec6)

    certs = [
        "IITG Statistics and EDA — Indian Institute of Technology, Guwahati",
        "IITG Introduction & Programming in Data Science — Indian Institute of Technology, Guwahati",
        "Adobe Photoshop Graphic Design — Adobe Certified"
    ]
    for c in certs:
        p_c = doc.add_paragraph(style='List Bullet')
        p_c.paragraph_format.space_before = Pt(1)
        p_c.paragraph_format.space_after = Pt(1)
        r_c = p_c.add_run(c)
        r_c.font.size = Pt(9.5)
        r_c.font.name = 'Calibri'

    doc.save(str(output_path))
    logger.info(f"DOCX saved to: {output_path}")
    return output_path


def generate_all_exports(
    resume_data: Dict[str, Any],
    optimized_resume: Dict[str, Any],
    user_edits: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Orchestrates the export workflow:
    1. Merges human review edits
    2. Synthesizes LaTeX code from base template
    3. Compiles LaTeX to PDF (using pdflatex)
    4. Writes raw LaTeX source (.tex)
    5. Generates beautifully formatted ATS DOCX
    6. Returns relative download URLs and status messages
    """
    merged_optimized = dict(optimized_resume)
    if user_edits:
        if isinstance(user_edits, dict):
            edits = user_edits.get("enhanced_bullets")
        else:
            edits = getattr(user_edits, "enhanced_bullets", None)
            
        if edits:
            merged_optimized["enhanced_bullets"] = edits

    sid, session_dir = create_session_dir(session_id)
    
    # 1. Generate Company-Ready DOCX
    docx_filename = "optimized_resume.docx"
    docx_path = session_dir / docx_filename
    generate_docx_document(resume_data, merged_optimized, docx_path)
    
    # 2. Generate LaTeX Source (Guaranteed Non-Empty)
    tex_content = generate_latex_source(resume_data, merged_optimized, api_key=api_key)
    
    # 3. Compile to PDF
    pdf_path = compile_latex_to_pdf(tex_content, session_dir, "optimized_resume")
    
    has_pdf = pdf_path is not None and pdf_path.exists()
    
    result = {
        "session_id": sid,
        "docx_url": f"/api/download/{sid}/{docx_filename}",
        "tex_url": f"/api/download/{sid}/optimized_resume.tex",
        "pdf_url": f"/api/download/{sid}/optimized_resume.pdf" if has_pdf else None,
        "latex_compiler_available": is_latex_compiler_available(),
        "message": (
            "Resume successfully compiled to PDF, LaTeX source (.tex), and Word (.docx)!"
            if has_pdf else
            "Company-ready Word (.docx) and LaTeX source (.tex) files ready. "
            "LaTeX source is 100% formatted and ready to compile on Overleaf or in Docker."
        )
    }
    return result
