# SPDX-License-Identifier: Apache-2.0
"""Deterministic coverage for #746: vllm_mlx_engine_steps_executed is never
populated for MLLM-routed models. Also covers the PR #749 review fix: a
step that raises must not be counted (mirrors AsyncEngineCore's
increment-after-success placement, engine_core.py).

The original #746 gap was two-layered:
- ``MLLMScheduler.get_stats()`` never produced a ``steps_executed`` key.
- ``BatchedEngine.get_stats()``'s MLLM promotion allowlist didn't forward
  that key to the top-level stats dict ``metrics.py`` reads even if it did.

These tests lock in both ends with fakes -- no real model/generation --
mirroring the fake/fixture style used in test_mllm_continuous_batching.py.
``vllm_mlx.mllm_scheduler`` and ``vllm_mlx.engine.batched`` both hard-import
``mlx.core`` at module scope. Real MLX is used where available; where it
isn't (e.g. the Linux CI matrix), ``tests._mlx_stub.install_if_unavailable``
stubs it so this import still succeeds -- the counter/promotion logic under
test is pure Python bookkeeping either way, so it's meaningful coverage in
both cases. Kept in its own CI step, separate from files with their own
real ``except ImportError`` mlx-optional handling -- see _mlx_stub.py's
docstring for why that matters.
"""

from tests import _mlx_stub

_mlx_stub.install_if_unavailable()

try:
    import mlx.core as mx  # noqa: F401

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

import pytest

pytestmark = pytest.mark.skipif(
    not HAS_MLX, reason="mlx.core not importable (even as a stub)"
)


class TestMLLMSchedulerStepsExecuted:
    """MLLMScheduler.get_stats() must report a step counter that increments
    once per step(), mirroring AsyncEngineCore._steps_executed
    (engine_core.py) for the plain-LLM path.
    """

    def _make_scheduler(self):
        from vllm_mlx.mllm_scheduler import MLLMScheduler

        class FakeTokenizer:
            eos_token_id = None

            def encode(self, text):
                return [1, 2, 3]

        class FakeProcessor:
            tokenizer = FakeTokenizer()

        class FakeModel:
            config = None

        scheduler = MLLMScheduler(model=FakeModel(), processor=FakeProcessor())
        # step() unconditionally calls _schedule_waiting(), which lazily
        # builds a real MLLMBatchGenerator from the (fake) model/processor.
        # The counter contract under test doesn't depend on that machinery,
        # so replace it with a no-op to keep this deterministic and fast.
        scheduler._schedule_waiting = lambda: []
        return scheduler

    def test_get_stats_reports_steps_executed(self):
        scheduler = self._make_scheduler()
        assert scheduler.get_stats()["steps_executed"] == 0

    def test_step_increments_steps_executed(self):
        scheduler = self._make_scheduler()

        scheduler.step()
        assert scheduler.get_stats()["steps_executed"] == 1

        scheduler.step()
        scheduler.step()
        assert scheduler.get_stats()["steps_executed"] == 3

    def test_step_that_raises_is_not_counted(self):
        """Regression for the PR #749 review: step() used to increment
        _steps_executed before any operation that can raise (schedule_waiting,
        the batch generator's forward pass, response processing), so a step
        that failed partway through -- and whose requests get terminated by
        _fail_requests_after_step_error rather than retried, since retrying a
        partially mutated batch is unsafe -- was still counted as executed.
        The counter must only advance on the successful-return path, mirroring
        AsyncEngineCore (engine_core.py), which increments only after
        self.scheduler.step() returns.
        """
        scheduler = self._make_scheduler()

        class ExplodingBatchGenerator:
            def process_pending_removals(self):
                pass

            def next(self):
                raise RuntimeError("simulated forward-pass failure")

        scheduler.batch_generator = ExplodingBatchGenerator()
        # Any truthy value satisfies step()'s `self.batch_generator is not
        # None and self.running` gate to reach next() -- get_stats() (which
        # would need a fully request-shaped fake here) is deliberately not
        # called in this test; the counter is checked directly below instead.
        scheduler.running = {"req-exploding": object()}

        with pytest.raises(RuntimeError, match="simulated forward-pass failure"):
            scheduler.step()

        assert scheduler._steps_executed == 0

        # The counter isn't left in a broken state by the earlier exception --
        # a subsequent step that actually completes still counts normally.
        scheduler.batch_generator = None
        scheduler.running = {}
        scheduler.step()
        assert scheduler.get_stats()["steps_executed"] == 1


class TestBatchedEngineStepsExecutedPromotion:
    """BatchedEngine.get_stats() must promote steps_executed from the MLLM
    scheduler's stats dict into the top-level stats dict metrics.py reads.
    """

    def _make_engine(self, mllm_stats):
        from vllm_mlx.engine.batched import BatchedEngine

        class FakeMLLMScheduler:
            def get_stats(self):
                return mllm_stats

        # get_stats() only touches a handful of plain attributes; bypass
        # BatchedEngine.__init__ (which loads a real model) and set just
        # those directly.
        engine = BatchedEngine.__new__(BatchedEngine)
        engine._mllm_scheduler = FakeMLLMScheduler()
        engine._engine = None
        engine._model_name = "fake-mllm-model"
        engine._created_at = 0.0
        engine._is_mllm = True
        engine._loaded = True
        engine._stream_interval = 1
        engine._mllm_draft_model = None
        return engine

    def test_promotes_steps_executed_to_top_level(self):
        engine = self._make_engine({"steps_executed": 42, "num_waiting": 0})

        stats = engine.get_stats()

        assert stats["steps_executed"] == 42

    def test_omits_steps_executed_when_scheduler_lacks_it(self):
        engine = self._make_engine({"num_waiting": 0})

        stats = engine.get_stats()

        assert "steps_executed" not in stats
