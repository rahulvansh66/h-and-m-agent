"""Judge models and adapters for RAGAS and DeepEval.

Role in the system: both libraries grade answers with an LLM. This module builds that judge from the project's
own Groq model (``evals.judge_model`` in the config, ``openai/gpt-oss-120b`` by default) so no other provider or
key is needed, and adapts it to each library's interface. Embeddings for RAGAS use the local fastembed model
(no API cost). It also switches off both libraries' telemetry. Judge calls are the expensive part of an
evaluation on the Groq free tier, so the model is configured with generous retries for rate limits.
"""

import asyncio
import os
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from typing import Any

# Set before the libraries are imported: neither should send usage data from this repo.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

from langchain_core.embeddings import Embeddings  # noqa: E402 - after the telemetry switches on purpose
from langchain_groq import ChatGroq  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from hm_agent.core.llm import load_model  # noqa: E402
from hm_agent.core.settings import EVAL_JUDGE_MODEL, RAG_EMBEDDING  # noqa: E402
from hm_agent.core.vector_store import get_embedder  # noqa: E402

# Groq free-tier 429s are retried with the client's backoff; evaluations are slow rather than failed.
JUDGE_MAX_RETRIES = 6

def judge_chat_model() -> ChatGroq:
    """Build the judge chat model with extra retries for rate limits.

    Returns:
        ChatGroq: Deterministic chat model of size ``evals.judge_model``.
    """
    model = load_model(EVAL_JUDGE_MODEL)
    model.max_retries = JUDGE_MAX_RETRIES
    return model


def run_sync(coro: Any) -> Any:
    """Run a coroutine to completion from sync code, even inside a running event loop.

    Args:
        coro (Any): The coroutine to run.

    Returns:
        Any: Its result.
    """
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def allow_ragas_import() -> None:
    """Provide the one module RAGAS imports but the installed ``langchain-community`` no longer ships.

    RAGAS imports ``ChatVertexAI`` at load time for its optional Vertex support. Newer ``langchain-community``
    removed that module, which stops RAGAS importing at all. A placeholder class is registered so the import
    succeeds; Vertex is never used here.
    """
    name = "langchain_community.chat_models.vertexai"
    if name in sys.modules:
        return
    try:
        __import__(name)
    except ModuleNotFoundError:
        stub = types.ModuleType(name)
        stub.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[name] = stub


def ragas_llm() -> Any:
    """Return the judge wrapped for RAGAS.

    Returns:
        Any: A ``LangchainLLMWrapper`` over the judge model.
    """
    allow_ragas_import()
    from ragas.llms import LangchainLLMWrapper
    from ragas.run_config import RunConfig

    return LangchainLLMWrapper(judge_chat_model(), run_config=RunConfig(timeout=180, max_retries=JUDGE_MAX_RETRIES, max_workers=1))


class LocalEmbeddings(Embeddings):
    """LangChain embeddings over the app's own cached fastembed model (the knowledge-base embedder)."""

    model = RAG_EMBEDDING  # RAGAS reads the model name from this attribute for its usage tracking

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed several texts.

        Args:
            texts (list[str]): Texts to embed.

        Returns:
            list[list[float]]: One vector per text.
        """
        return [[float(x) for x in vector] for vector in get_embedder(RAG_EMBEDDING).embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        """Embed one query.

        Args:
            text (str): Query text.

        Returns:
            list[float]: The query vector.
        """
        return [float(x) for x in next(iter(get_embedder(RAG_EMBEDDING).query_embed(text)))]


def ragas_embeddings() -> Any:
    """Return local fastembed embeddings wrapped for RAGAS (used by ``response_relevancy``).

    Returns:
        Any: A ``LangchainEmbeddingsWrapper`` over the knowledge-base embedding model (no API cost).
    """
    allow_ragas_import()
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(LocalEmbeddings())


def deepeval_judge() -> Any:
    """Return the judge adapted to DeepEval's model interface.

    Returns:
        Any: A ``DeepEvalBaseLLM`` that answers with the Groq judge, with structured output when DeepEval asks for a schema.
    """
    from deepeval.models import DeepEvalBaseLLM

    class GroqJudge(DeepEvalBaseLLM):
        """DeepEval adapter around the project's Groq chat model."""

        def __init__(self) -> None:
            """Build the underlying chat model."""
            self.model = judge_chat_model()

        def load_model(self) -> ChatGroq:
            """Return the chat model (DeepEval hook).

            Returns:
                ChatGroq: The judge model.
            """
            return self.model

        def generate(self, prompt: str, schema: type[BaseModel] | None = None) -> Any:
            """Answer a prompt, as the schema object when one is requested.

            Args:
                prompt (str): Judge prompt written by DeepEval.
                schema (type[BaseModel] | None): Pydantic schema DeepEval wants back.

            Returns:
                Any: The schema instance, or the reply text when no schema is given.
            """
            if schema is not None:  # json_schema mode: the default tool-calling mode makes gpt-oss call a bogus "json" tool
                return self.model.with_structured_output(schema, method="json_schema").invoke(prompt)
            return self.model.invoke(prompt).text

        async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None) -> Any:
            """Async variant of ``generate``.

            Args:
                prompt (str): Judge prompt written by DeepEval.
                schema (type[BaseModel] | None): Pydantic schema DeepEval wants back.

            Returns:
                Any: The schema instance, or the reply text when no schema is given.
            """
            if schema is not None:
                return await self.model.with_structured_output(schema, method="json_schema").ainvoke(prompt)
            return (await self.model.ainvoke(prompt)).text

        def get_model_name(self) -> str:
            """Return the judge model name (DeepEval hook).

            Returns:
                str: Groq model id.
            """
            return self.model.model_name

    return GroqJudge()
