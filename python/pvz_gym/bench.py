"""Register the PvZ task with latency-sensitive-bench when it is installed."""

from latency_bench.envs.adapters.gymnasium import GymnasiumEnvAdapter
from latency_bench.envs.factory import register_env_adapter


@register_env_adapter("pvz_day", env_id="PvZPortable/Day-v0", registration_imports=("pvz_gym.bench",))
def make_env(config, noop_action):
    env = config["env"]
    if env["env_fps"] != 100:
        raise ValueError("PvZ uses exactly 100 raw steps per second")
    return GymnasiumEnvAdapter(
        env_id="PvZPortable/Day-v0",
        registration_imports=["pvz_gym"],
        make_kwargs=env["make_kwargs"],
        env_fps=100,
        noop_action=noop_action,
        frame_stack=1,
        export_env_raw_rgb_frames=env["export_env_raw_rgb_frames"],
    )
