"""Long-lived task integration contracts owned by pvz_gym, with optional bench dependency."""

import numpy as np
import pytest

pytest.importorskip("latency_bench")
from latency_bench.core.types import Action
from latency_bench.envs.factory import resolve_env_config
from latency_bench.envs.latency_wrapper import ActionLatencyWrapper
from latency_bench.executors.env_step_backend import SubprocEnvStepBackend

import pvz_gym.bench
from pvz_gym import PvZEnv
from pvz_gym.validate import NOOP, trace_action


def test_subprocess_slots_use_the_gym_task_contract(native_kwargs):
    config = {"env": {
        "name": "pvz_day", "registration_imports": ["pvz_gym.bench"],
        "env_fps": 100, "obs_fps": 10, "export_env_raw_rgb_frames": False,
        "make_kwargs": native_kwargs,
    }}
    resolve_env_config(config)
    noop = Action(NOOP, is_noop=True)
    backend = SubprocEnvStepBackend(config=config, noop_action=noop, num_slots=2, start_method="spawn")
    try:
        a = backend.reset_slot(0, episode_id=0, seed=42)
        b = backend.reset_slot(1, episode_id=0, seed=42)
        np.testing.assert_array_equal(a.data, b.data)
        result = backend.step_slots({0: noop})[0].result
        assert result.info["sim_time_ms"] == 10
        observations = backend.observe_slots([0, 1])
        assert observations[0].env_step == 1 and observations[1].env_step == 0
        assert set(observations[0].metadata) == {"env_step", "sim_time_ms", "observation"}
        backend.reset_slot(0, episode_id=1, seed=1)
        unchanged = backend.observe_slots([1])[1]
        np.testing.assert_array_equal(b.data, unchanged.data)
    finally:
        backend.close()


@pytest.mark.parametrize("delay", [0, 2])
def test_declared_latency_matches_an_explicit_native_trace(native_kwargs, delay):
    raw = PvZEnv(**native_kwargs)
    wrapped = ActionLatencyWrapper(
        PvZEnv(**native_kwargs), noop_action=NOOP, env_fps=100, obs_fps=100,
        fixed_latency_raw_frames=delay, hold_policy="one_frame_then_noop", ordering_policy="issue_order_fifo",
    )
    try:
        raw.reset(seed=42)
        wrapped.reset(seed=42)
        # Release each input through the same public latency wrapper used in evaluation.
        for tick in range(1, 15):
            action = trace_action(tick)
            wrapped.step(action)
            raw.step(trace_action(tick - delay) if tick > delay else NOOP)
            assert raw.diagnostics() == wrapped.unwrapped.diagnostics(), tick
    finally:
        raw.close()
        wrapped.close()
