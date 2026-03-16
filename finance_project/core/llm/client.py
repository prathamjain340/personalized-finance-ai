# LLM interface
# app/core/llm/client.py

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "500"))
_TIMEOUT_SECONDS = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "15"))


def _token_budget(env_key: str, default_value: int) -> int:
    raw_value = os.getenv(env_key, str(default_value))
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        parsed = default_value
    return min(_MAX_TOKENS, max(64, parsed))


_FINANCE_RESPONSE_MAX_TOKENS = _token_budget("OPENAI_FINANCE_RESPONSE_MAX_TOKENS", 320)
_FINANCE_VOICE_MAX_TOKENS = _token_budget("OPENAI_FINANCE_VOICE_MAX_TOKENS", 180)
_CLARIFICATION_MAX_TOKENS = _token_budget("OPENAI_CLARIFICATION_MAX_TOKENS", 220)
_TURN_CONTROL_MAX_TOKENS = _token_budget("OPENAI_TURN_CONTROL_MAX_TOKENS", 192)

_OP_MAX_TOKENS = {
    "turn_control": _TURN_CONTROL_MAX_TOKENS,
    "out_of_domain_response": 180,
    "clarification_response": _CLARIFICATION_MAX_TOKENS,
    "finance_response": _FINANCE_RESPONSE_MAX_TOKENS,
    "finance_response_voice": _FINANCE_VOICE_MAX_TOKENS,
    "live_data_response": 180,
    "voice_render": min(_MAX_TOKENS, 260),
}
_OP_TEMPERATURE = {
    "turn_control": 0.1,
    "finance_response_voice": 0.25,
    "voice_render": 0.2,
}


def generate_response(prompt: str, operation: str = "llm_generation") -> str:
    """
    Stateless LLM interface.

    Responsibilities:
    - Send fully assembled prompt to the LLM
    - Return generated text only
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt is empty")

    max_tokens = _OP_MAX_TOKENS.get(operation, _MAX_TOKENS)
    temperature = _OP_TEMPERATURE.get(operation, _TEMPERATURE)

    try:
        response = _client.chat.completions.create(
            model=_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": prompt}],
            timeout=_TIMEOUT_SECONDS,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "I am having trouble processing this request right now. Please try again in a moment."
