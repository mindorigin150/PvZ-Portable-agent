"""Long-lived Gym contracts owned by pvz_gym: clocks, inputs, reset and cleanup."""

import os
import time

import gymnasium as gym
import numpy as np
import pytest

from pvz_gym import PvZEnv
from pvz_gym.validate import NOOP, trace_action


def test_mouse_clock_and_repeatable_reset(native_kwargs):
    with PvZEnv(**native_kwargs) as env:
        initial, info = env.reset(seed=42)
        assert initial.shape == (600, 800, 3) and initial.dtype == np.uint8
        assert info == {"env_step": 0, "sim_time_ms": 0}
        state = env.diagnostics()
        np.testing.assert_array_equal(initial, env.render())
        assert state == env.diagnostics()
        for tick in range(1, 6):
            _, _, _, _, info = env.step(trace_action(tick))
            assert info == {"env_step": tick, "sim_time_ms": tick * 10}
        planted = env.diagnostics()
        assert len(planted["plants"]) == 1
        assert planted["sun"] == state["sun"] - 50
        assert planted["cards"][0][3] == 1
        time.sleep(0.05)
        assert planted == env.diagnostics()
        reset_image, _ = env.reset(seed=42)
        np.testing.assert_array_equal(initial, reset_image)
        assert env.diagnostics() == state
        with pytest.raises(ValueError):
            env.step((800, 0, 0))
        with pytest.raises(ValueError):
            env.step((0.5, 0, 0))
    with pytest.raises(gym.error.ResetNeeded):
        env.step(NOOP)


def test_truncation_failure_and_process_cleanup(native_kwargs):
    env = PvZEnv(**native_kwargs, max_episode_steps=1)
    try:
        env.reset(seed=0)
        pid = env._process.pid
        _, reward, terminated, truncated, _ = env.step(NOOP)
        assert reward == 0 and not terminated and truncated
        with pytest.raises(gym.error.ResetNeeded):
            env.step(NOOP)
    finally:
        env.close()
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    with PvZEnv(**native_kwargs) as env:
        env.reset(seed=0)
        env._process.kill()
        env._process.wait()
        with pytest.raises((RuntimeError, BrokenPipeError)):
            env.step(NOOP)
        assert env._process is None


def test_complete_loss_episode(native_kwargs):
    with PvZEnv(**native_kwargs) as env, PvZEnv(**native_kwargs) as replay:
        env.reset(seed=0)
        replay.reset(seed=0)
        for tick in range(60_000):
            transition = env.advance(NOOP)
            assert transition == replay.advance(NOOP)
            reward, terminated, truncated, _ = transition
            if tick % 100 == 0 or terminated:
                assert env.diagnostics() == replay.diagnostics()
                np.testing.assert_array_equal(env.render(), replay.render())
            if terminated or truncated:
                break
        assert terminated and not truncated and reward == -1
