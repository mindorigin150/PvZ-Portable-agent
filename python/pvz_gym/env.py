"""Own one native worker and expose RGB observations with literal mouse inputs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import select
import struct
import subprocess
import tempfile
import time

import gymnasium as gym
import numpy as np

_REQUEST = struct.Struct("<4I")
_RESPONSE = struct.Struct("<4sIiIII")
_RGB_BYTES = 800 * 600 * 3


class PvZEnv(gym.Env):
    """Run fixed day level 1-8 in an isolated engine process at 100 Hz."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    def __init__(
        self,
        *,
        binary: str,
        resource_dir: str,
        render_mode: str = "rgb_array",
        max_episode_steps: int = 90_000,
        timeout: float = 120.0,
        reference: bool = False,
    ):
        super().__init__()
        self.binary = Path(binary).resolve(strict=True)
        self.resource_dir = Path(resource_dir).resolve(strict=True)
        if not (self.resource_dir / "main.pak").is_file():
            raise FileNotFoundError(self.resource_dir / "main.pak")
        if not (self.resource_dir / "properties").is_dir():
            raise FileNotFoundError(self.resource_dir / "properties")
        if render_mode != "rgb_array":
            raise ValueError("render_mode must be rgb_array")
        if max_episode_steps <= 0 or timeout <= 0:
            raise ValueError("max_episode_steps and timeout must be positive")
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.timeout = timeout
        self.reference = reference
        self.action_space = gym.spaces.MultiDiscrete([800, 600, 6], dtype=np.int64)
        self.observation_space = gym.spaces.Box(0, 255, (600, 800, 3), dtype=np.uint8)
        self._process = None
        self._save = None
        self._log = None
        self._frame = None
        self._tick = 0
        self._ended = True

    def _read(self, count: int, deadline: float) -> bytes:
        result = bytearray()
        while len(result) < count:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self._process.stdout], [], [], remaining)[0]:
                raise TimeoutError(f"PvZ worker response exceeded {self.timeout:g} seconds")
            data = os.read(self._process.stdout.fileno(), count - len(result))
            if not data:
                self._log.seek(0)
                raise RuntimeError("PvZ worker closed its pipe:\n" + self._log.read().decode(errors="replace")[-6000:])
            result.extend(data)
        return bytes(result)

    def _receive(self, *, pixels: bool, diagnostic: bool = False, stepping: bool = False):
        deadline = time.monotonic() + self.timeout
        magic, tick, reward, outcome, image_bytes, state_bytes = _RESPONSE.unpack(self._read(_RESPONSE.size, deadline))
        if magic != b"PVZG" or outcome > 2 or image_bytes != (_RGB_BYTES if pixels else 0):
            raise RuntimeError("Invalid PvZ worker response header")
        if state_bytes > 2_000_000 or bool(state_bytes) != diagnostic:
            raise RuntimeError("Invalid PvZ worker diagnostic response")
        expected_reward = {0: 0, 1: 1, 2: -1}[outcome] if stepping else 0
        if tick != self._tick + int(stepping) or reward != expected_reward:
            raise RuntimeError("PvZ worker violated its tick or reward contract")
        if pixels:
            self._frame = np.frombuffer(self._read(image_bytes, deadline), dtype=np.uint8).reshape(600, 800, 3).copy()
        state = json.loads(self._read(state_bytes, deadline)) if diagnostic else None
        self._tick = tick
        info = {"env_step": tick, "sim_time_ms": tick * 10}
        return reward, bool(outcome), info, state

    def _request(self, op: int, action=(0, 0, 0)):
        if self._process is None:
            raise gym.error.ResetNeeded("Call reset before using the environment")
        if op in (1, 4):
            action = np.asarray(action)
            if action.dtype.kind not in "iu" or not self.action_space.contains(action):
                raise ValueError(f"Action must contain integers in {self.action_space}")
        try:
            self._process.stdin.write(_REQUEST.pack(op, *(int(v) for v in action)))
            self._process.stdin.flush()
            return self._receive(pixels=op in (1, 2), diagnostic=op == 3, stepping=op in (1, 4))
        except BaseException:
            self.close()
            raise

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options:
            raise ValueError("The fixed day-level fixture does not accept reset options")
        if seed is not None and not 0 <= seed <= 0xFFFFFFFF:
            raise ValueError("The native seed must fit an unsigned 32-bit integer")
        native_seed = seed if seed is not None else int(self.np_random.integers(0, 2**32))
        self.close()
        self._save = tempfile.TemporaryDirectory(prefix="pvz-gym-")
        self._log = tempfile.TemporaryFile()
        try:
            self._process = subprocess.Popen(
                [str(self.binary), "--gym-worker", str(native_seed), str(int(self.reference)),
                 f"-resdir={self.resource_dir}", f"-savedir={self._save.name}"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                cwd=self._save.name, bufsize=0,
            )
            _, _, info, _ = self._receive(pixels=True)
        except BaseException:
            self.close()
            raise
        self._ended = False
        return self._frame.copy(), info

    def step(self, action):
        if self._ended:
            raise gym.error.ResetNeeded("Reset is required before stepping")
        reward, terminated, info, _ = self._request(1, action)
        truncated = not terminated and self._tick >= self.max_episode_steps
        self._ended = terminated or truncated
        return self._frame.copy(), float(reward), terminated, truncated, info

    def render(self):
        self._request(2)
        return self._frame.copy()

    def diagnostics(self):
        """Return privileged evaluator state. Never include it in actor input."""
        return self._request(3)[3]

    def advance(self, action):
        """Advance a diagnostic trace without transporting an RGB image."""
        if self._ended:
            raise gym.error.ResetNeeded("Reset is required before stepping")
        reward, terminated, info, _ = self._request(4, action)
        truncated = not terminated and self._tick >= self.max_episode_steps
        self._ended = terminated or truncated
        return float(reward), terminated, truncated, info

    def close(self):
        process, self._process = self._process, None
        if process is not None:
            try:
                if process.poll() is None:
                    try:
                        process.stdin.write(_REQUEST.pack(0, 0, 0, 0))
                        process.stdin.flush()
                    except (BrokenPipeError, OSError):
                        pass
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            finally:
                process.stdin.close()
                process.stdout.close()
        if self._log is not None:
            self._log.close()
            self._log = None
        if self._save is not None:
            self._save.cleanup()
            self._save = None
        self._frame = None
        self._tick = 0
        self._ended = True
