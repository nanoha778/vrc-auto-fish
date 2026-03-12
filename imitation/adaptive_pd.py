# adaptive_pd.py
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

import config


@dataclass
class AdaptivePDDecision:
    press: bool
    probability: float
    kp: float
    kd: float
    delta_kp: float
    delta_kd: float
    hold: float
    mode: str


class GainAdapterNet(nn.Module):
    FEATURES_PER_FRAME = 10

    def __init__(self, history_len: int = 10, hidden: int = 128):
        super().__init__()
        self.history_len = history_len
        inp = history_len * self.FEATURES_PER_FRAME
        self.net = nn.Sequential(
            nn.Linear(inp, hidden),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
            nn.Tanh(),  # ΔKp, ΔKd を -1~1 に制限
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdaptivePDController:
    FEATURES_PER_FRAME = 10

    def __init__(
        self,
        model_path: Optional[str] = None,
        history_len: Optional[int] = None,
    ):
        self.model_path = model_path or getattr(
            config,
            "ADAPTIVE_PD_MODEL_PATH",
            os.path.join(config.BASE_DIR, "imitation", "adaptive_pd.pt"),
        )
        self.history_len = history_len or getattr(config, "ADAPTIVE_PD_HISTORY_LEN", 10)

        self._window = deque(maxlen=self.history_len)
        self._model: Optional[GainAdapterNet] = None
        self._norm_mean: Optional[np.ndarray] = None
        self._norm_std: Optional[np.ndarray] = None
        self._loaded = False
        self._enabled = False

        self._load()

    def reset(self):
        self._window.clear()

    @property
    def enabled(self) -> bool:
        return self._enabled and self._loaded and self._model is not None

    def _load(self):
        if not os.path.exists(self.model_path):
            self._enabled = False
            self._loaded = False
            return

        ckpt = None

        # 1) まず安全寄りに weights_only=True を試す
        try:
            ckpt = torch.load(self.model_path, map_location="cpu", weights_only=True)
        except TypeError:
            # 古い torch 互換
            ckpt = torch.load(self.model_path, map_location="cpu")
        except Exception as e:
            # numpy 配列入り checkpoint だとここに来ることがある
            print(f"[AdaptivePD] weights_only=True load failed, fallback to full load: {e}")
            ckpt = torch.load(self.model_path, map_location="cpu", weights_only=False)

        history_len = int(ckpt.get("history_len", self.history_len))
        self.history_len = history_len
        self._window = deque(maxlen=self.history_len)

        self._model = GainAdapterNet(history_len=self.history_len)
        self._model.load_state_dict(ckpt["model_state"])
        self._model.eval()

        norm_mean = ckpt.get("norm_mean")
        norm_std = ckpt.get("norm_std")
        self._norm_mean = np.array(norm_mean, dtype=np.float32) if norm_mean is not None else None
        self._norm_std = np.array(norm_std, dtype=np.float32) if norm_std is not None else None

        self._loaded = True
        self._enabled = True

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
        feats = np.array([
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
        self._window.append(feats)

    def ready(self) -> bool:
        return len(self._window) >= max(2, self.history_len)

    def _build_input(self) -> Optional[np.ndarray]:
        if not self.ready():
            return None

        arr = list(self._window)
        x = np.concatenate(arr, axis=0).astype(np.float32)

        if self._norm_mean is not None and self._norm_std is not None:
            std = self._norm_std.copy()
            std[std < 1e-6] = 1.0
            x = (x - self._norm_mean) / std

        return x

    @torch.no_grad()
    def decide(
        self,
        error: float,
        velocity: float,
        base_hold: float,
        min_hold: float,
        max_hold: float,
        hold_gain: float,
        speed_damping: float,
        delta_kp_scale: float = 0.030,
        delta_kd_scale: float = 0.0025,
        threshold_margin: float = 0.001,
    ) -> AdaptivePDDecision:
        """
        既存PD:
            hold = base_hold + abs(error)*hold_gain + velocity*speed_damping

        学習後:
            kp = hold_gain + ΔKp
            kd = speed_damping + ΔKd
        """
        kp = hold_gain
        kd = speed_damping
        delta_kp = 0.0
        delta_kd = 0.0
        mode = "fallback"

        if self.enabled and self.ready():
            x = self._build_input()
            if x is not None:
                xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                out = self._model(xt).squeeze(0).cpu().numpy()

                delta_kp = float(out[0]) * delta_kp_scale
                delta_kd = float(out[1]) * delta_kd_scale

                kp = max(0.0, hold_gain + delta_kp)
                kd = speed_damping + delta_kd
                mode = "adaptive"

        hold = base_hold + abs(error) * kp + velocity * kd
        hold = max(min_hold, min(max_hold, hold))

        # いまの bot 側互換: hold が最小より十分大きければ press 扱い
        press = hold >= (min_hold + threshold_margin)

        # 確率っぽい表示用。厳密な確率ではなく 0~1 に圧縮したスコア
        score = (hold - min_hold) / max(1e-6, (max_hold - min_hold))
        probability = float(max(0.0, min(1.0, score)))

        return AdaptivePDDecision(
            press=press,
            probability=probability,
            kp=kp,
            kd=kd,
            delta_kp=delta_kp,
            delta_kd=delta_kd,
            hold=hold,
            mode=mode,
        )