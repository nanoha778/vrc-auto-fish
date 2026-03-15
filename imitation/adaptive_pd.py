# adaptive_pd.py
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Optional

import numpy as np


@dataclass
class AdaptivePDDecision:
    press: bool
    hold: float
    kp: float
    kd: float
    delta_kp: float
    delta_kd: float
    mode: str


class AdaptivePDController:
    def __init__(
        self,
        history_len: int = 10,
        base_kp: float = 0.040,
        base_kd: float = 0.00025,
        min_hold: float = 0.015,
        max_hold: float = 0.100,
    ):
        self.history_len = history_len
        self.base_kp = base_kp
        self.base_kd = base_kd
        self.min_hold = min_hold
        self.max_hold = max_hold

        self.window = deque(maxlen=history_len)
        self.policy = None  # SB3 model を後から注入

    def reset(self):
        self.window.clear()

    def set_policy(self, policy):
        self.policy = policy

    def update_features(
        self,
        error: float,
        velocity: float,
        bar_h: float,
        fish_delta: float,
        dist_ratio: float,
        mouse_prev: float,
        fish_in_bar: float,
        press_streak: float,
        predicted: float,
        bar_accel: float,
    ):
        obs = np.array([
            error,
            velocity,
            bar_h,
            fish_delta,
            dist_ratio,
            mouse_prev,
            fish_in_bar,
            press_streak,
            predicted,
            bar_accel,
        ], dtype=np.float32)
        self.window.append(obs)

    def ready(self) -> bool:
        return len(self.window) == self.history_len

    def build_obs(self) -> Optional[np.ndarray]:
        if not self.ready():
            return None
        return np.concatenate(list(self.window), axis=0).astype(np.float32)

    def decide(self):
        obs = self.build_obs()
        if obs is None or self.policy is None:
            return AdaptivePDDecision(
                press=False,
                hold=self.min_hold,
                kp=self.base_kp,
                kd=self.base_kd,
                delta_kp=0.0,
                delta_kd=0.0,
                mode="fallback",
            )

        action, _ = self.policy.predict(obs, deterministic=True)
        delta_kp = float(action[0])
        delta_kd = float(action[1])

        kp = max(0.0, self.base_kp + delta_kp)
        kd = self.base_kd + delta_kd

        error = obs[-10]
        velocity = obs[-9]

        hold = self.min_hold + abs(error) * kp + velocity * kd
        hold = float(np.clip(hold, self.min_hold, self.max_hold))
        press = hold > (self.min_hold + 1e-3)

        return AdaptivePDDecision(
            press=press,
            hold=hold,
            kp=kp,
            kd=kd,
            delta_kp=delta_kp,
            delta_kd=delta_kd,
            mode="rl",
        )