import os
import time
import shutil
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from config import (
    HOST,
    PORT,
    BACKEND_DIR,
    WEB_APP_DIR,
    PROJECT_ROOT,
    FRONTEND_DIR,
    TEMP_DIR,
    TEMP_FILE_TTL_MINUTES,
)
from routes.optimize import router as optimize_router
from routes.export import router as export_router

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ai_resume_optimizer")


def cleanup_expired_temp_files():
    """Removes temporary export directories older than TEMP_FILE_TTL_MINUTES."""
    if not TEMP_DIR.exists():
        return
    now = time.time()
    cutoff = now - (TEMP_FILE_TTL_MINUTES * 60)
    
    cleaned_count = 0
    for entry in TEMP_DIR.iterdir():
        try:
            if entry.is_dir() and entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
                cleaned_count += 1
            elif entry.is_file() and entry.stat().st_mtime < cutoff:
                entry.unlink(missing_ok=True)
                cleaned_count += 1
        except Exception as e:
            logger.warning(f"Could not clean up temp path {entry}: {e}")
            
    if cleaned_count > 0:
        logger.info(f"Cleaned up {cleaned_count} expired temp directories/files.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup and shutdown procedures."""
    logger.info("Starting AI Resume Optimizer Backend Server...")
    cleanup_expired_temp_files()
    yield
    logger.info("Shutting down AI Resume Optimizer Backend Server...")
    cleanup_expired_temp_files()


app = FastAPI(
    title="AI Resume Optimizer API",
    version="1.0.0",
    description="Production-ready FastAPI backend for ATS resume optimization using LangGraph agents.",
    lifespan=lifespan,
)

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(optimize_router)
app.include_router(export_router)

# Helper for resolving frontend files across deployment contexts
def find_frontend_file(filename: str):
    """Finds a frontend asset across candidate paths."""
    candidates = [
        FRONTEND_DIR / filename if FRONTEND_DIR else None,
        BACKEND_DIR / "frontend" / filename,
        BACKEND_DIR / "FrontEnd" / filename,
        WEB_APP_DIR / "FrontEnd" / filename,
        PROJECT_ROOT / "Web App" / "FrontEnd" / filename,
        Path.cwd() / "frontend" / filename,
        Path.cwd() / "FrontEnd" / filename,
        BACKEND_DIR / filename,
    ]
    for c in candidates:
        if c and c.exists():
            return c
    return None


# Mount static asset directory
static_dir = FRONTEND_DIR if (FRONTEND_DIR and FRONTEND_DIR.exists()) else (BACKEND_DIR / "frontend")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=FileResponse)
async def serve_index():
    """Serves the main application UI."""
    ui_file = find_frontend_file("UI.html")
    if ui_file:
        return FileResponse(str(ui_file))
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"message": "UI.html not found in any frontend candidate path."}
    )


@app.get("/UI.html", response_class=FileResponse)
async def serve_ui_html():
    """Direct access to UI.html."""
    ui_file = find_frontend_file("UI.html")
    if ui_file:
        return FileResponse(str(ui_file))
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"message": "UI.html not found."}
    )


@app.get("/logo.png", response_class=FileResponse)
async def serve_logo():
    """Serves the application logo."""
    logo_file = find_frontend_file("logo.png")
    if logo_file:
        return FileResponse(str(logo_file), media_type="image/png")
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"message": "logo.png not found."}
    )


@app.get("/logo_white.png", response_class=FileResponse)
async def serve_logo_white():
    """Serves the white application logo."""
    logo_file = find_frontend_file("logo_white.png")
    if logo_file:
        return FileResponse(str(logo_file), media_type="image/png")
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"message": "logo_white.png not found."}
    )


@app.get("/favicon.ico", response_class=FileResponse)
async def serve_favicon():
    """Serves the favicon in white format."""
    logo_file = find_frontend_file("logo_white.png") or find_frontend_file("logo.png")
    if logo_file:
        return FileResponse(str(logo_file), media_type="image/png")
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"message": "favicon not found."}
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all unhandled exception handler to return clean JSON errors."""
    logger.exception(f"Unhandled error on {request.method} {request.url.path}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "error": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
