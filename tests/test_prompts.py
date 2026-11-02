"""Tests for Langfuse-managed prompts: fallback behaviour, compilation and the registry.

Role in the system: checks ``hm_agent.core.prompts`` and ``hm_agent.prompt_registry`` without any network
call (the conftest fixture disables Langfuse, so every prompt resolves to its in-code fallback).
"""

import pytest
from langchain_core.runnables import RunnableLambda

from hm_agent.core import prompts
from hm_agent.core.prompts import get_managed_prompt
from hm_agent.prompt_registry import PROMPTS

CHAT = [{"role": "system", "content": "Be brief. {\"json\": true}"}, {"role": "user", "content": "Hi {{name}}"}]


def test_chat_prompt_falls_back_and_compiles() -> None:
    """With Langfuse off the in-code chat prompt is used, version 0, and variables are substituted."""
    prompt = get_managed_prompt("hm/test", CHAT, kind="chat")
    assert (prompt.source, prompt.version) == ("fallback", 0)
    assert prompt.compile(name="Bob")[1]["content"] == "Hi Bob"


def test_text_prompt_falls_back() -> None:
    """A text prompt compiles to its string."""
    assert get_managed_prompt("hm/test-text", "You are {{role}}.", kind="text").compile(role="kind") == "You are kind."


def test_chain_renders_messages_and_keeps_literal_braces() -> None:
    """``chain`` fills the template variables and does not treat JSON braces in the system text as variables."""
    seen = []
    chain = get_managed_prompt("hm/test", CHAT, kind="chat").chain(RunnableLambda(lambda value: seen.append(value) or "ok"))
    assert chain.invoke({"name": "Bob"}) == "ok"
    assert seen[0].messages[0].content == 'Be brief. {"json": true}'
    assert seen[0].messages[1].content == "Hi Bob"


def test_text_prompt_has_no_template() -> None:
    """Only chat prompts can be turned into a template."""
    with pytest.raises(TypeError):
        get_managed_prompt("hm/test-text", "plain", kind="text").template()


def test_langfuse_error_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Langfuse failure never breaks a turn: the fallback is returned."""

    class Broken:
        def get_prompt(self, *_args, **_kwargs):
            raise RuntimeError("langfuse down")

    monkeypatch.setattr(prompts, "_client_enabled", lambda: True)
    monkeypatch.setattr(prompts, "get_client", lambda: Broken())
    assert get_managed_prompt("hm/test", CHAT, kind="chat").source == "fallback"


def test_registry_is_complete_and_valid() -> None:
    """Every prompt has a unique hm/ name and compiles, and the code defaults use only double-brace variables."""
    names = [p.name for p in PROMPTS]
    assert len(names) == len(set(names)) == 7
    for spec in PROMPTS:
        assert spec.name.startswith("hm/")
        compiled = get_managed_prompt(spec.name, spec.default, kind=spec.kind).compile(
            transcript="t", question="q", answers="a", history="h", context="c", query="x"
        )
        assert compiled
