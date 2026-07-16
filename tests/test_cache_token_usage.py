# SPDX-License-Identifier: Apache-2.0
"""Tests for surfacing prefix-cache reuse in the OpenAI-compatible usage.

The prefix cache already computes how many prompt tokens are served from the
KV/prefix cache; these tests lock in that the value is threaded all the way to
the client under ``usage.prompt_tokens_details.cached_tokens`` for both the
non-streaming and (the historically broken) streaming chat completion paths.
"""

import json

import pytest

from vllm_mlx.api.models import PromptTokensDetails, Usage
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.output_collector import RequestOutputCollector
from vllm_mlx.request import Request, RequestOutput, SamplingParams


class TestCachedTokensDataModel:
    """Field plumbing through the data model layer."""

    def test_usage_prompt_tokens_details_defaults_none(self):
        assert Usage().prompt_tokens_details is None

    def test_usage_serializes_cached_tokens(self):
        usage = Usage(
            prompt_tokens=10,
            completion_tokens=2,
            total_tokens=12,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=7),
        )
        assert usage.model_dump()["prompt_tokens_details"] == {"cached_tokens": 7}

    def test_prompt_tokens_details_default_cached_zero(self):
        assert PromptTokensDetails().cached_tokens == 0

    def test_request_output_carries_cached_tokens(self):
        out = RequestOutput(
            request_id="r", prompt_tokens=10, completion_tokens=2, cached_tokens=7
        )
        assert out.cached_tokens == 7
        # The OpenAI-compatible usage dict exposes it too.
        assert out.usage["cached_tokens"] == 7

    def test_generation_output_cached_tokens_default(self):
        assert GenerationOutput(text="x").cached_tokens == 0

    def test_request_tracks_peak_cached_tokens(self):
        req = Request(
            request_id="r",
            prompt="hi",
            sampling_params=SamplingParams(),
            cached_tokens=12,
            peak_cached_tokens=12,
        )
        assert req.peak_cached_tokens == 12


class TestOutputCollectorMerge:
    """Aggregated streaming output must not drop the cache-reuse count."""

    def test_merge_keeps_max_cached_tokens(self):
        # Simulate the producer getting ahead of the consumer: the early step
        # carries the prefix-cache reuse, a later step reports 0. The merged
        # output the consumer eventually reads must retain the peak.
        collector = RequestOutputCollector(aggregate=True)
        collector.put(
            RequestOutput(
                request_id="r", new_text="a", completion_tokens=1, cached_tokens=9
            )
        )
        collector.put(
            RequestOutput(
                request_id="r",
                new_text="b",
                completion_tokens=2,
                cached_tokens=0,
                finished=True,
            )
        )
        merged = collector.get_nowait()
        assert merged is not None
        assert merged.finished is True
        assert merged.cached_tokens == 9


class TestGetUsage:
    """The shared streaming usage builder surfaces cache reuse."""

    def test_get_usage_surfaces_cached_tokens(self):
        from vllm_mlx.server import get_usage

        usage = get_usage(
            GenerationOutput(
                text="x", prompt_tokens=10, completion_tokens=2, cached_tokens=7
            )
        )
        assert usage.prompt_tokens == 10
        assert usage.prompt_tokens_details is not None
        assert usage.prompt_tokens_details.cached_tokens == 7

    def test_get_usage_zero_cache(self):
        from vllm_mlx.server import get_usage

        usage = get_usage(
            GenerationOutput(text="x", prompt_tokens=10, completion_tokens=2)
        )
        assert usage.prompt_tokens_details.cached_tokens == 0


class TestChatCompletionCachedTokens:
    """End-to-end: cached_tokens reaches the client on both paths."""

    @pytest.mark.anyio
    async def test_non_streaming_chat_surfaces_cached_tokens(self, monkeypatch):
        from vllm_mlx.server import (
            ChatCompletionRequest,
            Message,
            create_chat_completion,
        )
        import vllm_mlx.server as server

        class FakeEngine:
            model_name = "fake-engine"
            is_mllm = False
            preserve_native_tool_format = False

            async def chat(self, messages, **kwargs):
                return GenerationOutput(
                    text="ok",
                    prompt_tokens=20,
                    completion_tokens=4,
                    cached_tokens=15,
                    finish_reason="stop",
                )

        fake_engine = FakeEngine()

        async def fake_acquire(raw_request, **kwargs):
            return fake_engine

        async def fake_release(**kwargs):
            return None

        monkeypatch.setattr(server, "_validate_model_name", lambda _m: None)
        monkeypatch.setattr(server, "_acquire_default_engine_for_request", fake_acquire)
        monkeypatch.setattr(server, "_release_default_engine", fake_release)
        monkeypatch.setattr(server, "_model_name", "served-model")
        monkeypatch.setattr(server, "_default_max_tokens", 128)
        monkeypatch.setattr(server, "_default_timeout", 30.0)
        monkeypatch.setattr(server, "_enable_auto_tool_choice", False)
        monkeypatch.setattr(server, "_tool_call_parser", None)
        monkeypatch.setattr(server, "_tool_parser_instance", None)

        request = ChatCompletionRequest(
            model="served-model",
            messages=[Message(role="user", content="Hello")],
            max_tokens=16,
        )

        response = await create_chat_completion(request, raw_request=None)

        assert response.usage.prompt_tokens == 20
        assert response.usage.prompt_tokens_details is not None
        assert response.usage.prompt_tokens_details.cached_tokens == 15

    @pytest.mark.anyio
    async def test_streaming_chat_surfaces_cached_tokens(self, monkeypatch):
        """The finished streaming chunk must report cache reuse.

        This is the path that previously lost the value: the streamed chunks
        are consumed one at a time, so the finished chunk itself has to carry
        the preserved peak rather than relying on aggregation.
        """
        from vllm_mlx.server import (
            ChatCompletionRequest,
            Message,
            stream_chat_completion,
        )
        import vllm_mlx.server as server

        class FakeEngine:
            model_name = "fake-engine"

            async def stream_chat(self, messages, **kwargs):
                # Intermediate chunks stream text; the finished chunk carries the
                # preserved prefix-cache reuse count.
                yield GenerationOutput(
                    text="Hel", new_text="Hel", finished=False, cached_tokens=15
                )
                yield GenerationOutput(
                    text="Hello",
                    new_text="lo",
                    finished=True,
                    finish_reason="stop",
                    prompt_tokens=20,
                    completion_tokens=2,
                    cached_tokens=15,
                )

        monkeypatch.setattr(server, "_model_name", "served-model")
        monkeypatch.setattr(server, "_reasoning_parser", None)
        monkeypatch.setattr(server, "_enable_auto_tool_choice", False)
        monkeypatch.setattr(server, "_tool_call_parser", None)
        monkeypatch.setattr(server, "_tool_parser_instance", None)

        request = ChatCompletionRequest(
            model="served-model",
            messages=[Message(role="user", content="hi")],
            stream=True,
        )

        chunks = [
            chunk
            async for chunk in stream_chat_completion(
                FakeEngine(), request.messages, request
            )
        ]
        payloads = [
            json.loads(chunk.removeprefix("data: ").strip())
            for chunk in chunks
            if chunk != "data: [DONE]\n\n"
        ]

        usages = [p["usage"] for p in payloads if p.get("usage")]
        assert usages, "expected at least one chunk carrying usage"
        # Every usage-bearing chunk reports the cache reuse.
        for usage in usages:
            assert usage["prompt_tokens_details"] == {"cached_tokens": 15}
