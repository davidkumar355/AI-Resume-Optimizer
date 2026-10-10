import os
from pathlib import Path

# Paths
BACKEND_DIR = Path(__file__).resolve().parent
WEB_APP_DIR = BACKEND_DIR.parent
PROJECT_ROOT = WEB_APP_DIR.parent

# Resolve FRONTEND_DIR: search backend/frontend first, then sibling and parent directories
FRONTEND_CANDIDATES = [
    BACKEND_DIR / "frontend",
    BACKEND_DIR / "FrontEnd",
    WEB_APP_DIR / "FrontEnd",
    PROJECT_ROOT / "Web App" / "FrontEnd",
    PROJECT_ROOT / "FrontEnd",
    Path.cwd() / "frontend",
    Path.cwd() / "FrontEnd",
]

FRONTEND_DIR = BACKEND_DIR / "frontend"
for candidate in FRONTEND_CANDIDATES:
    if candidate.exists() and (candidate / "UI.html").exists():
        FRONTEND_DIR = candidate
        break

TEMP_DIR = BACKEND_DIR / "temp"

# Ensure temp directory exists
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# LaTeX template resolution: check backend first, then project root
LATEX_TEMPLATE_NAME = "resume laytex format.tex"
LATEX_TEMPLATE_PATH = BACKEND_DIR / LATEX_TEMPLATE_NAME
if not LATEX_TEMPLATE_PATH.exists():
    LATEX_TEMPLATE_PATH = PROJECT_ROOT / LATEX_TEMPLATE_NAME

# Server configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# File upload guardrails
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".pdf"}

# Rate limiting
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX", "5"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

# Temp file cleanup TTL
TEMP_FILE_TTL_MINUTES = int(os.getenv("TEMP_FILE_TTL_MINUTES", "30"))

# Model configuration for Groq
HIGH_TIER_MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
]

LOW_TIER_MODELS = [
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
    "openai/gpt-oss-120b",
    "llama-3.1-8b-instant",
]
