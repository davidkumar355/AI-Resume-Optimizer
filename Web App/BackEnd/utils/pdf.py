import os
import re
from typing import Union
import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_source: Union[str, bytes]) -> str:
    """
    Extracts text cleanly from a PDF file path or raw PDF bytes.
    Preserves document structure while normalizing whitespace.
    """
    if isinstance(pdf_source, str):
        if not os.path.exists(pdf_source):
            raise FileNotFoundError(f"PDF file not found at: {pdf_source}")
        doc = fitz.open(pdf_source)
    elif isinstance(pdf_source, bytes):
        if not pdf_source:
            raise ValueError("Provided PDF byte buffer is empty.")
        doc = fitz.open(stream=pdf_source, filetype="pdf")
    else:
        raise TypeError("pdf_source must be a file path string or bytes.")

    try:
        full_text = ""
        for page_num in range(len(doc)):
            page = doc[page_num]
            full_text += page.get_text("text") + "\n"

        # Clean excessive newlines while preserving paragraph structure
        full_text = re.sub(r'\n{3,}', '\n\n', full_text)
        full_text = full_text.strip()

        if not full_text:
            raise ValueError("Unable to extract text from PDF. The document might be scanned/image-only or encrypted.")

        return full_text
    finally:
        doc.close()
