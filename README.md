# 🚀 AI-Powered Resume-to-JD Gap Analyzer & Builder

An interactive web application designed to analyze the gap between your resume and a target Job Description (JD), calculate ATS (Applicant Tracking System) scores before and after optimization, and generate a fully tailored, ATS-friendly LaTeX resume ready for download or import into Overleaf.

Powered by **LangGraph**, **Groq LLM** (`llama-3.3-70b-versatile`), and **PyMuPDF**.

---

## 🌟 Features

1. **Structured Resume Parsing:** Extracts skills, projects, work experience, education, and certifications from your uploaded PDF resume.
2. **Job Description Extraction:** Identifies required/preferred skills, responsibilities, and experience requirements from pasted JDs.
3. **ATS Scoring Engine:** Calculates a weighted ATS match score before and after optimization across 5 dimensions (Skill Match, Keyword Coverage, Experience Alignment, Resume Quality, Project Relevance).
4. **Gap Analysis:** Pinpoints exact matched/missing skills, missing keywords, and weak areas.
5. **Resume Enhancement Agent:** Rewrites weak bullet points using the professional **STAR/XYZ** format without fabricating credentials.
6. **ATS LaTeX Builder:** Generates an ATS-optimized LaTeX resume, integrating missing keywords and enhanced bullet points into your LaTeX template.
7. **Local PDF Compilation:** Automatically compiles the LaTeX code to a downloadable PDF if a local TeX distribution is detected. If not, it provides clean Overleaf integration instructions.
8. **Interactive Streamlit Dashboard:** Shows a real-time progress logger of the AI agent pipeline execution flow, side-by-side metric comparisons, and download areas.

---

## 🛠️ Tech Stack
- **Inference Engine:** Groq API (`llama-3.3-70b-versatile`)
- **Agentic Orchestration:** LangGraph
- **Framework integration:** LangChain
- **Frontend Dashboard:** Streamlit
- **PDF Extraction:** PyMuPDF (`fitz`)
- **Data Validation:** Pydantic v2
- **Document Output:** LaTeX / TeX

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- A Groq API Key (get one for free at [console.groq.com](https://console.groq.com/))

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/davidkumar355/AI-Resume-Optimizer.git
   cd AI-Resume-Optimizer
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory and add your Groq API Key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. Place your starting LaTeX template as `resume laytex format.tex` in the root directory (a sample is included).

---

## 🖥️ Running the Application

Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser to run the app.

---

## 📝 License
This project is open-source and available under the [MIT License](LICENSE).
