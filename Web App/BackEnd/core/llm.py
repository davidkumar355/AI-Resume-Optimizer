import os
from typing import Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from config import HIGH_TIER_MODELS, LOW_TIER_MODELS

load_dotenv()


def get_llm_with_fallback(task_ambiguity: str = "low", api_key: Optional[str] = None):
    """
    Returns a ChatGroq LLM instance with automated fallback models.
    Supports BYOK (Bring Your Own Key) by accepting an explicit api_key.
    Falls back to the environment GROQ_API_KEY if not provided.
    """
    resolved_key = (api_key or "").strip() or os.getenv("GROQ_API_KEY", "").strip()
    
    if not resolved_key or resolved_key == "YOUR_GROQ_API_KEY_HERE":
        raise ValueError(
            "Groq API key not found. Please provide your Groq API key via the UI settings "
            "or set the GROQ_API_KEY environment variable."
        )

    if task_ambiguity == "high":
        models_to_use = HIGH_TIER_MODELS
        temperature = 0.5
    else:
        models_to_use = LOW_TIER_MODELS
        temperature = 0.0

    llms = [
        ChatGroq(
            model=model,
            api_key=resolved_key,
            temperature=temperature,
            max_tokens=6000,
            max_retries=2,
        )
        for model in models_to_use
    ]

    primary_llm = llms[0]
    return primary_llm.with_fallbacks(llms[1:])
