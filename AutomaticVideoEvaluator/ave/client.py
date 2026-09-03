from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Config, api_settings


@dataclass
class ModelResult:
    json: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0


class ModelClient:
    def __init__(self, config: Config):
        from openai import OpenAI

        key, base_url = api_settings()
        kwargs: dict[str, Any] = {
            "api_key": key,
            "base_url": base_url,
            "timeout": config.request_timeout_seconds,
            # AVE owns the retry loop so token/cost accounting remains explicit.
            "max_retries": 0,
        }
        self.client = OpenAI(**kwargs)
        self.config = config

    def json_completion(
        self, model: str, system: str, content: list[dict[str, Any]]
    ) -> ModelResult:
        from json_repair import repair_json

        request = dict(
            model=model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
        )
        try:
            response = self.client.chat.completions.create(**request)
        except Exception as error:
            message = str(error).lower()
            if "response_format" not in message and "json_object" not in message:
                raise
            # Some OpenAI-compatible Seed gateways return JSON correctly but do
            # not advertise the optional response_format request parameter.
            request.pop("response_format")
            response = self.client.chat.completions.create(**request)
        raw = response.choices[0].message.content or "{}"
        import json

        parsed = json.loads(repair_json(raw, ensure_ascii=False))
        usage = response.usage
        return ModelResult(
            json=parsed,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
