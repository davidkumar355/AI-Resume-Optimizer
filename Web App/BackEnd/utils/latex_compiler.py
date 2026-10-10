import os
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ai_resume_optimizer.latex")


def get_pdflatex_path() -> Optional[str]:
    """
    Locates the pdflatex executable across Linux and Windows systems.
    """
    path = shutil.which("pdflatex")
    if path:
        return path

    # Windows fallback locations (MiKTeX and TeX Live standard install locations)
    windows_paths = [
        r"C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe",
        r"C:\Program Files (x86)\MiKTeX\miktex\bin\x64\pdflatex.exe",
        r"C:\texlive\2026\bin\windows\pdflatex.exe",
        r"C:\texlive\2025\bin\windows\pdflatex.exe",
        r"C:\texlive\2024\bin\windows\pdflatex.exe",
        r"C:\texlive\2023\bin\windows\pdflatex.exe",
    ]
    for p in windows_paths:
        if os.path.exists(p):
            return p

    return None


def is_latex_compiler_available() -> bool:
    """Returns True if a valid LaTeX compiler (pdflatex) is detected on the host."""
    return get_pdflatex_path() is not None


def compile_latex_to_pdf(
    tex_content: str,
    output_dir: Path,
    output_filename: str = "optimized_resume",
    timeout_seconds: int = 30
) -> Optional[Path]:
    """
    Compiles raw LaTeX source code into a PDF file in output_dir.
    Always writes the .tex file regardless of compiler availability.
    Returns the Path to the compiled .pdf if successful, or None if unavailable/failed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tex_file = output_dir / f"{output_filename}.tex"
    with open(tex_file, "w", encoding="utf-8") as f:
        f.write(tex_content)
        
    logger.info(f"LaTeX source written to: {tex_file}")

    pdflatex_path = get_pdflatex_path()
    if not pdflatex_path:
        logger.warning(
            "pdflatex is not installed on this host. PDF compilation skipped. "
            "The raw .tex file is available for download."
        )
        return None

    try:
        # Run pdflatex non-interactively
        # Running once or twice resolves labels and geometry
        for run_idx in range(2):
            result = subprocess.run(
                [
                    pdflatex_path,
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    f"-output-directory={str(output_dir)}",
                    str(tex_file)
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_seconds,
                check=False
            )
            if result.returncode != 0:
                logger.error(f"pdflatex failed on pass {run_idx + 1}: {result.stdout[:500]}")
                break

        pdf_path = output_dir / f"{output_filename}.pdf"
        if pdf_path.exists():
            logger.info(f"PDF successfully compiled to: {pdf_path}")
            return pdf_path

    except subprocess.TimeoutExpired:
        logger.error("LaTeX compilation timed out.")
    except Exception as e:
        logger.error(f"LaTeX compilation error: {e}")

    return None
