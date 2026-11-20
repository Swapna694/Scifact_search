"""The only file that talks to the LLM (Google Gemini API).

To switch to another provider (Groq, Ollama, Claude...), change only this file.
"""
import time

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_FALLBACK_MODEL, GEMINI_MODEL

_client = None
RETRIES_PER_MODEL = 3
WAIT_SECONDS = 5

# shared client
def _get_client():
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _is_temporary(error):
    msg = str(error)
    return any(code in msg for code in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE"))

# calling in the model sends your prompt to Gemini and returns the answer as text
def ask_llm(prompt, system=None, temperature=0.2):
    """Send a prompt and return the model's text answer ('' if nothing came back)."""
    config = types.GenerateContentConfig(system_instruction=system, temperature=temperature)

    #Make the list of models to try so that same model isn't tried for twice   ["gemini-3.5-flash", "gemini-2.5-flash"]
    models = [GEMINI_MODEL]
    if GEMINI_FALLBACK_MODEL and GEMINI_FALLBACK_MODEL != GEMINI_MODEL:
        models.append(GEMINI_FALLBACK_MODEL)

    last_error = None
    #For each model, it makes up to 3 attempts
    for model in models:
        for attempt in range(RETRIES_PER_MODEL):
            try:
                resp = _get_client().models.generate_content(
                    model=model, contents=prompt, config=config
                )
                return (resp.text or "").strip()
            except Exception as e:
                last_error = e
                if not _is_temporary(e):
                    raise
                if attempt < RETRIES_PER_MODEL - 1:
                    wait = WAIT_SECONDS * (attempt + 1)
                    print(f"({model} busy, retrying in {wait}s...)")
                    time.sleep(wait)
        print(f"({model} still unavailable, trying the next model...)")
    raise last_error