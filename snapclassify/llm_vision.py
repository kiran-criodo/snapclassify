"""Ask Gemini for the breed of one photo: the multimodal-LLM side of the comparison. Given.

The reply is constrained to a JSON object whose "breed" must be one of the class names (response_json_schema with
an enum), which makes it parseable almost always; a reply that still can't be read counts as wrong. Only dataset
photos are sent: the free tier may keep what you send, and shelter photos need a contract first.
"""
from __future__ import annotations

import io
import json
import os
import time

from PIL import Image

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
PRICES = {"input": 0.75, "output": 3.75}     # USD per 1M tokens, gemini-3.8-flash list price (checked 27 Sep 2026)
PROMPT = "Which breed is the pet in this photo? Answer with exactly one breed from the allowed list."


def make_client():
    from google import genai
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def classify_photo(client, image: Image.Image, class_names: list[str], model: str = MODEL) -> dict:
    """{"label": a class name or None, "latency_s", "input_tokens", "output_tokens", "error"} for one photo."""
    from google.genai import types
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    schema = {"type": "object", "properties": {"breed": {"type": "string", "enum": list(class_names)}},
              "required": ["breed"]}
    config = types.GenerateContentConfig(
        response_mime_type="application/json", response_json_schema=schema,
        thinking_config=types.ThinkingConfig(thinking_level="low"),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
    t0 = time.perf_counter()
    try:
        r = client.models.generate_content(
            model=model, contents=[types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"), PROMPT],
            config=config)
    except Exception as e:  # noqa: BLE001 - one failed call is a wrong answer here, not a crash
        return {"label": None, "latency_s": round(time.perf_counter() - t0, 3), "input_tokens": 0,
                "output_tokens": 0, "error": f"{type(e).__name__}: {str(e)[:120]}"}
    latency = round(time.perf_counter() - t0, 3)
    u = r.usage_metadata
    tokens_in = (u.prompt_token_count or 0) if u else 0
    tokens_out = ((u.candidates_token_count or 0) + (u.thoughts_token_count or 0)) if u else 0
    try:
        label = json.loads(r.text or "").get("breed")
    except (json.JSONDecodeError, AttributeError):
        label = None
    return {"label": label if label in class_names else None, "latency_s": latency, "input_tokens": tokens_in,
            "output_tokens": tokens_out, "error": None}
