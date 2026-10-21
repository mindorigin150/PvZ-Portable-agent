"""Gymnasium registration for the fixed PvZ day-level benchmark."""

from gymnasium.envs.registration import register

from pvz_gym.env import PvZEnv

register(id="PvZPortable/Day-v0", entry_point="pvz_gym.env:PvZEnv")

__all__ = ["PvZEnv"]
