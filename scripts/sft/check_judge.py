#!/usr/bin/env python3
"""Fail fast when the LongVT LLM judge is misconfigured or unauthorized."""

from __future__ import annotations

import json
import os
import re
import sys

from openai import OpenAI


def main() -> int:
    api_key = os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("JUDGE_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    model = os.getenv("JUDGE_MODEL_NAME") or os.getenv("OPENAI_MODEL_NAME")
    if not api_key or not base_url or not model:
        print("Judge preflight failed: API key, base URL, or model is unset.")
        return 1

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Return 1 if these answers match semantically, otherwise 0. "
                    "Output only one digit.\n"
                    "Ground truth: blue jeans\nPrediction: blue denim trousers"
                ),
            }
        ],
        "max_tokens": int(os.getenv("JUDGE_MAX_TOKENS", "8")),
        "temperature": 0,
    }
    extra_body = os.getenv("JUDGE_EXTRA_BODY", "").strip()
    if extra_body:
        payload["extra_body"] = json.loads(extra_body)

    try:
        response = OpenAI(api_key=api_key, base_url=base_url).chat.completions.create(
            **payload
        )
    except Exception as error:  # noqa: BLE001
        print(f"Judge preflight failed: {type(error).__name__}: {error}")
        return 1

    raw = (response.choices[0].message.content or "").strip()
    if re.fullmatch(r"[01]", raw) is None:
        print(f"Judge preflight failed: expected 0 or 1, received {raw!r}")
        return 1

    print(f"Judge preflight passed: {model} returned {raw}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
