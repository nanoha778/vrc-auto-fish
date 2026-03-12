# train_adaptive_pd.py
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO


class FishingPDEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, simulator, history_len: int = 10):
        super().__init__()
        self.simulator = simulator
        self.history_len = history_len
        self.feat_dim = 10

        obs_dim = history_len * self.feat_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # delta_kp, delta_kd
        self.action_space = spaces.Box(
            low=np.array([-0.03, -0.0025], dtype=np.float32),
            high=np.array([0.03,  0.0025], dtype=np.float32),
            dtype=np.float32,
        )

        self.base_kp = 0.040
        self.base_kd = 0.00025
        self.hold_min = 0.015
        self.hold_max = 0.100

        self.window = None
        self.prev_press = 0.0

    def _get_obs(self):
        return np.concatenate(self.window, axis=0).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.simulator.reset()
        self.prev_press = 0.0

        first = self.simulator.get_features(prev_press=self.prev_press)
        self.window = [first.copy() for _ in range(self.history_len)]
        return self._get_obs(), {}

    def step(self, action):
        delta_kp = float(action[0])
        delta_kd = float(action[1])

        obs = self._get_obs()
        last = obs[-10:]
        error = float(last[0])
        velocity = float(last[1])

        kp = max(0.0, self.base_kp + delta_kp)
        kd = self.base_kd + delta_kd

        hold = self.hold_min + abs(error) * kp + velocity * kd
        hold = float(np.clip(hold, self.hold_min, self.hold_max))
        press = 1.0 if hold > self.hold_min + 1e-3 else 0.0

        next_features, outcome = self.simulator.step(press=press, hold=hold)

        self.window.pop(0)
        self.window.append(next_features)

        fish_in_bar = float(next_features[6])
        abs_error = abs(float(next_features[0]))
        toggle_penalty = abs(press - self.prev_press)

        reward = (
            0.03 * fish_in_bar
            - 0.02 * abs_error
            - 0.01 * (abs(delta_kp) + abs(delta_kd))
            - 0.005 * toggle_penalty
        )

        terminated = False
        truncated = False

        if outcome == "success":
            reward += 5.0
            terminated = True
        elif outcome == "escape":
            reward -= 3.0
            terminated = True
        elif outcome == "line_break":
            reward -= 5.0
            terminated = True

        self.prev_press = press
        return self._get_obs(), reward, terminated, truncated, {}

def main():
    simulator = ...  # ここを既存の釣りロジック/ログ再生器に差し込む
    env = FishingPDEnv(simulator=simulator, history_len=10)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.003,
        clip_range=0.2,
        tensorboard_log="./runs/adaptive_pd_rl/",
    )
    model.learn(total_timesteps=1_000_000)
    model.save("adaptive_pd_rl")