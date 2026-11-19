"""The only file that talks to the LLM (Google Gemini API).

To switch to another provider (Groq, Ollama, Claude...), change only this file.
"""
import time

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL

_client = None


def _get_client():
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def ask_llm(prompt, system=None, temperature=0.2):
    """Send a prompt and return the model's text answer ('' if nothing came back)."""
    config = types.GenerateContentConfig(system_instruction=system, temperature=temperature)
    for attempt in range(3):
        try:
            resp = _get_client().models.generate_content(
                model=GEMINI_MODEL, contents=prompt, config=config
            )
            return (resp.text or "").strip()
        except Exception as e:
            rate_limited = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
            if rate_limited and attempt < 2:
                print(f"(rate limited, retrying in {10 * (attempt + 1)}s...)")
                time.sleep(10 * (attempt + 1))
                continue
            raise