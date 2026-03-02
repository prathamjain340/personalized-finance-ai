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

_OP_MAX_TOKENS = {
    "turn_control": 160,
    "out_of_domain_response": 180,
    "clarification_response": 180,
    "finance_response": min(_MAX_TOKENS, 240),
    "live_data_response": 180,
}
_OP_TEMPERATURE = {
    "turn_control": 0.1,
}


def generate_response(
    prompt: str,
    operation: str = "llm_generation",
    max_tokens_override: int | None = None,
) -> str:
    """
    Stateless LLM interface.

    Responsibilities:
    - Send fully assembled prompt to the LLM
    - Return generated text only
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt is empty")

    if isinstance(max_tokens_override, int) and max_tokens_override > 0:
        max_tokens = max_tokens_override
    else:
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
