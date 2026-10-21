"""Compare native execution and measure independent RGB environment slots."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import platform
import subprocess
import time

from gymnasium.utils.env_checker import check_env
import numpy as np

from pvz_gym.env import PvZEnv

NOOP = (0, 0, 0)


def trace_action(tick):
    """Replay fixed mouse events. This trace does not inspect the game state."""
    return {
        1: (110, 45, 2), 2: (110, 45, 3),
        3: (120, 130, 2), 4: (120, 130, 3),
        8: (170, 45, 2), 9: (170, 45, 3),
        10: (200, 230, 2), 11: (200, 230, 3),
        12: (200, 230, 4), 13: (200, 230, 5),
    }.get(tick, NOOP)


def compare(kwargs, seeds, ticks):
    results = []
    for seed in seeds:
        with PvZEnv(**kwargs) as native, PvZEnv(**kwargs, reference=True) as client:
            native.reset(seed=seed)
            client.reset(seed=seed)
            for tick in range(ticks + 1):
                a, b = native.diagnostics(), client.diagnostics()
                if a != b:
                    fields = [key for key in a.keys() | b.keys() if a.get(key) != b.get(key)]
                    raise AssertionError(f"First client divergence: seed={seed}, tick={tick}, fields={fields}")
                if tick % 100 == 0:
                    if not np.array_equal(native.render(), client.render()):
                        raise AssertionError(f"First client pixel divergence: seed={seed}, tick={tick}")
                if tick == ticks:
                    break
                action = trace_action(tick + 1)
                if native.advance(action) != client.advance(action):
                    raise AssertionError(f"Client transition mismatch: seed={seed}, tick={tick + 1}")
            results.append({"seed": seed, "matched_ticks": ticks})
    return results


def process_rss(pid):
    # Linux deployment measurements include the native process, not just its wrapper.
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError(f"VmRSS unavailable for worker {pid}")


def benchmark(kwargs, slots, ticks):
    envs = [PvZEnv(**kwargs) for _ in range(slots)]
    try:
        with ThreadPoolExecutor(max_workers=slots) as pool:
            started = time.perf_counter()
            list(pool.map(lambda e: e.reset(seed=42), envs))
            startup = time.perf_counter() - started
            # All image transport and renderer costs remain in the measured path.
            started = time.perf_counter()
            for tick in range(1, ticks + 1):
                action = trace_action(tick)
                transitions = list(pool.map(lambda e: e.step(action), envs))
                if any(t[2] or t[3] for t in transitions):
                    raise RuntimeError("Benchmark episode ended before the requested sample completed")
                if any(not np.array_equal(transitions[0][0], t[0]) for t in transitions[1:]):
                    raise AssertionError(f"Batch pixel mismatch at tick {tick}")
            elapsed = time.perf_counter() - started
            rss = sum(process_rss(env._process.pid) for env in envs)
            before = envs[-1].diagnostics()
            envs[0].reset(seed=1)
            if slots > 1 and envs[-1].diagnostics() != before:
                raise AssertionError("Reset changed another slot")
            return {
                "slots": slots, "ticks_per_slot": ticks, "startup_seconds": startup,
                "elapsed_seconds": elapsed, "aggregate_ticks_per_second": slots * ticks / elapsed,
                "ticks_per_second_per_slot": ticks / elapsed, "native_rss_bytes": rss,
                "rgb_bytes_per_tick_per_slot": 800 * 600 * 3,
            }
    finally:
        for env in envs:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--resource-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--compare-ticks", type=int, default=3600)
    parser.add_argument("--benchmark-ticks", type=int, default=100)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 42])
    parser.add_argument("--slots", nargs="+", type=int, default=[1, 4, 16])
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    kwargs = {"binary": args.binary, "resource_dir": args.resource_dir}
    source = Path(__file__).resolve().parents[2]
    report = {
        "evidence": "native engine validation, no model calls",
        "host": platform.node(), "platform": platform.platform(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
        "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True)),
        "resource_dir": str(Path(args.resource_dir).resolve()),
        "env_fps": 100, "obs_fps_example": 10,
        "trace": {str(t): trace_action(t) for t in range(1, 14)},
    }
    with PvZEnv(**kwargs) as env:
        check_env(env)
        image, _ = env.reset(seed=42)
        from PIL import Image  # Only the validation CLI needs image export.
        Image.fromarray(image).save(output / "initial.png")
    report["gym_checker"] = "passed"
    print("Gym checker passed", flush=True)
    report["client_comparison"] = compare(kwargs, args.seeds, args.compare_ticks)
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Client comparison passed", flush=True)
    report["batch"] = []
    for slots in args.slots:
        result = benchmark(kwargs, slots, args.benchmark_ticks)
        report["batch"].append(result)
        (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
