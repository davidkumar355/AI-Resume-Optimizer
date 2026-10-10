import json
import re
from typing import Any


def parse_json_from_response(response_text: str) -> Any:
    """
    Robust JSON parser for LLM outputs.
    Handles reasoning tags, markdown fences, surrounding text, and trailing commas.
    """
    text = response_text.strip()
    
    if not text:
        raise ValueError("Could not extract JSON from LLM response: Response is completely empty.")

    # 1. Strip reasoning tags (<think>...</think>)
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    elif "<think>" in text:
        first_brace = text.find("{")
        if first_brace != -1:
            text = text[first_brace:].strip()
        else:
            raise ValueError(
                "LLM ran out of output tokens while thinking (unclosed <think> tag). "
                "Output token limit should be increased."
            )

    # 2. Extract from markdown code fences if present
    json_fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if json_fence_match:
        fence_content = json_fence_match.group(1).strip()
        try:
            return json.loads(fence_content)
        except json.JSONDecodeError:
            text = fence_content

    # 3. Direct JSON parsing
    try:
        return json.loads(text)
    except Exception:
        pass
        
    # 4. Extract substring between first '{' and last '}'
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_str = text[start_idx:end_idx + 1]
        try:
            return json.loads(json_str)
        except Exception:
            # Fix common trailing comma issues
            json_str_fixed = re.sub(r',\s*([\}\]])', r'\1', json_str)
            try:
                return json.loads(json_str_fixed)
            except Exception:
                pass

    # 5. Extract substring between first '[' and last ']'
    start_idx = text.find('[')
    end_idx = text.rfind(']')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_str = text[start_idx:end_idx + 1]
        try:
            return json.loads(json_str)
        except Exception:
            json_str_fixed = re.sub(r',\s*([\}\]])', r'\1', json_str)
            try:
                return json.loads(json_str_fixed)
            except Exception:
                pass

    raise ValueError(f"Could not extract JSON from LLM response:\n{response_text}")
