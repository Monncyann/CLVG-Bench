from types import SimpleNamespace

import openai

from ave.client import ModelClient
from ave.config import DEFAULT_ARK_BASE_URL, Config


def test_seed_client_defaults_to_ark_timeout_and_json_fallback(monkeypatch) -> None:
    calls = []
    constructor = {}

    class Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            if "response_format" in kwargs:
                raise ValueError("response_format is unsupported")
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))
                ],
                usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
            )

    def fake_openai(**kwargs):
        constructor.update(kwargs)
        return SimpleNamespace(
            chat=SimpleNamespace(completions=Completions()),
        )

    monkeypatch.setenv("AVE_API_KEY", "test-key")
    monkeypatch.delenv("AVE_BASE_URL", raising=False)
    monkeypatch.setattr(openai, "OpenAI", fake_openai)
    config = Config(request_timeout_seconds=123.0)
    client = ModelClient(config)
    result = client.json_completion("model", "system", [{"type": "text", "text": "x"}])

    assert constructor == {
        "api_key": "test-key",
        "base_url": DEFAULT_ARK_BASE_URL,
        "timeout": 123.0,
        "max_retries": 0,
    }
    assert len(calls) == 2
    assert "response_format" not in calls[1]
    assert result.json == {"ok": True}
    assert (result.input_tokens, result.output_tokens) == (3, 2)
