# PvZ Gymnasium environment

`PvZPortable/Day-v0` uses the original renderer and game logic from this fork.
It runs level 1-8 with a fresh synthetic profile and a fixed six-card loadout:
Sunflower, Peashooter, Cherry Bomb, Wall-nut, Potato Mine, and Snow Pea.
Reset completes the normal intro before returning the first battle image.
The Gym wrapper currently supports Linux. Particle shake uses stable particle
IDs, so process addresses do not alter visual replay.

## Build and run

Use a separate Git worktree. On Debian or Ubuntu, install the native build
dependencies listed in the main README, plus `xvfb`, `xauth`, and Mesa for a
server with no desktop. Build the normal executable:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPVZ_DEBUG=OFF
cmake --build build -j 12
python3 -m venv /ABSOLUTE/PATH/pvz-venv
/ABSOLUTE/PATH/pvz-venv/bin/pip install -e '.[test]'
```

Provide explicit paths to the built executable and a resource directory.
The resource directory must contain `main.pak` and `properties/` from a compatible
GOTY resource package. Resources stay outside this repository.

```python
import gymnasium as gym
import pvz_gym

with gym.make(
    "PvZPortable/Day-v0",
    binary="/ABSOLUTE/PATH/build/pvz-portable",
    resource_dir="/ABSOLUTE/PATH/resources",
) as env:
    rgb, info = env.reset(seed=42)
    rgb, reward, terminated, truncated, info = env.step([110, 45, 2])
    rgb, reward, terminated, truncated, info = env.step([110, 45, 3])
```

Run the Python program inside a virtual display on a server:

```bash
xvfb-run -a -s '-screen 0 1024x768x24 -nolisten tcp' \
  env LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=1 SDL_AUDIODRIVER=dummy \
  /ABSOLUTE/PATH/pvz-venv/bin/python your_program.py
```

This setup uses CPU software rendering. Each native worker owns one OpenGL
context. Temporary displays do not accept TCP connections.

## Observation, action, and clock contracts

| Item | Contract |
| --- | --- |
| Observation | `uint8[600, 800, 3]`, RGB, top row first |
| Action | `MultiDiscrete([800, 600, 6])`, ordered as `(x, y, event)` |
| Event IDs | 0 noop, 1 move, 2 left down, 3 left up, 4 right down, 5 right up |
| Native step | Exactly one logic tick, 10 ms, 100 Hz |
| Input timing | Apply the input before the tick |
| Noop | Ignore coordinates and preserve the cursor and button state |
| Button state | Retain until an explicit release event |
| Reward | +1 when the level award appears, -1 when zombies win, otherwise 0 |
| Termination | First victory or defeat tick |
| Truncation | 90,000 ticks by default, configurable with `max_episode_steps` |
| Public info | `env_step`, `sim_time_ms` |

`render()` does not advance the game. Time spent waiting for a model does not
advance this synchronous Gym interface. A latency executor must supply the
waiting ticks. The normal client remains available through its original entry
point without `--gym-worker`.

`reset(seed=...)` restarts the native process in a fresh temporary save directory.
The fixed profile ID, progress, loadout, and RNG seed define the initial state.
The wrapper accepts unsigned 32-bit seeds and rejects unknown reset options.
`close()` waits for the worker and releases its pipes and temporary files.

`diagnostics()` exposes privileged evaluator state, including RNG state, sun,
cooldowns, and entity state. It never appears in observations or `info`.
`advance(action)` is for diagnostic traces that do not need an RGB transfer.
Neither method belongs in an online visual actor's tool surface.

## Batch environments and latency-sensitive-bench

Each `PvZEnv` owns a separate native process. Multiple environments can run in
the same Python process or in `gymnasium.vector.AsyncVectorEnv` workers.
The native engine globals never cross environment processes.

Install this package into every interpreter that constructs the task.
Import `pvz_gym.bench` to register task name `pvz_day` with
latency-sensitive-bench. This module reuses `GymnasiumEnvAdapter`.
Set these task fields:

```yaml
env:
  name: pvz_day
  registration_imports: [pvz_gym.bench]
  env_fps: 100
  obs_fps: 10
  frame_stack: 1
  export_env_raw_rgb_frames: false
  make_kwargs:
    binary: /ABSOLUTE/PATH/build/pvz-portable
    resource_dir: /ABSOLUTE/PATH/resources
```

Native mouse actions are integer triples. Pass them through the task's policy
adapter without converting them into plant commands. Use `[0, 0, 0]` as the
native noop and `one_frame_then_noop` for one-shot mouse events.
Use `SubprocEnvStepBackend` for batched evaluation and retain the existing
executor's ordering and latency semantics. The generic native MultiDiscrete
interface does not add support to training policies that accept only Box or
Discrete actions.

## Validation

The validation CLI runs from this source checkout, including editable installs.
It checks Gymnasium, compares the worker with the original client's clock
scheduler, and measures 1, 4, and 16 independent environments.
Both paths use the same fixture. The reference path advances with
`LawnApp::UpdateAppStep`; the worker advances the same full logical update chain
without wall-clock pacing. This comparison checks this fork's execution paths.
It does not compare this reimplementation against the original Windows game.

```bash
xvfb-run -a -s '-screen 0 1024x768x24 -nolisten tcp' \
  env LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=1 SDL_AUDIODRIVER=dummy \
  /ABSOLUTE/PATH/pvz-venv/bin/pvz-validate \
  --binary /ABSOLUTE/PATH/build/pvz-portable \
  --resource-dir /ABSOLUTE/PATH/resources \
  --output-dir /ABSOLUTE/PATH/outputs/pvz-validation
```

The fixed mouse trace plants a Sunflower and attempts an unaffordable purchase.
The comparison checks every tick and reports the first differing state field.
The batch measurement includes rendering and raw RGB transfer on every step.
It checks equal-seed pixels and independent resets.
Reports distinguish native engine validation from model evaluation.
Record the resource source and version beside each report.

Long-lived integration tests live in `python/tests` and require the same display:

```bash
/ABSOLUTE/PATH/pvz-venv/bin/python -m pytest python/tests \
  --binary /ABSOLUTE/PATH/build/pvz-portable \
  --resource-dir /ABSOLUTE/PATH/resources
```

The tests protect mouse actions, seed replay, clocks, truncation, complete loss,
worker failure, and cleanup. They skip explicitly if resource paths are absent.
