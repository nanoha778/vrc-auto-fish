# imitation/hybrid_policy.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

import config
from imitation.model import FishPolicy


@dataclass
class HybridDecision:
    """ハイブリッド制御の出力"""
    press: bool
    probability: float
    mode: str              # "pd", "assist_press", "assist_release", "fallback"
    confidence: float      # 0~1


class HybridPolicyController:
    """
    PD制御 + imitation のハイブリッド制御器。

    方針:
    - 明確な場面は PD を優先
    - 迷う場面だけ imitation で補助
    - imitation の確率が強く偏ったときだけ PD を上書き
    """

    FEATURES_PER_FRAME = 10

    def __init__(
        self,
        model_path: Optional[str] = None,
        history_len: Optional[int] = None,
        assist_band: float = 0.18,
        strong_press: float = 0.72,
        strong_release: float = 0.28,
        min_history_ratio: float = 0.8,
    ):
        self.model_path = model_path or config.IL_MODEL_PATH
        self.history_len = history_len or config.IL_HISTORY_LEN

        # PDが迷う領域でだけ assist するための帯域
        self.assist_band = assist_band

        # 強く上書きする閾値
        self.strong_press = strong_press
        self.strong_release = strong_release

        # 何フレーム以上たまったら推論するか
        self.min_history = max(1, int(self.history_len * min_history_ratio))

        self._window = deque(maxlen=self.history_len)
        self._loaded = False
        self._enabled = False

        self._model: Optional[FishPolicy] = None
        self._norm_mean: Optional[np.ndarray] = None
        self._norm_std: Optional[np.ndarray] = None
        self._device = "cpu"

        self._load()

    def _load(self):
        """policy.pt をロード（旧形式/新形式の両対応）"""
        if not os.path.exists(self.model_path):
            self._enabled = False
            self._loaded = False
            return

        try:
            ckpt = torch.load(self.model_path, map_location="cpu", weights_only=True)
        except TypeError:
            # 古い torch 互換
            ckpt = torch.load(self.model_path, map_location="cpu")

        # --------------------------------------------------
        # 保存形式の互換対応
        # 新形式:
        #   {
        #     "model_state": ...,
        #     "norm_mean": ...,
        #     "norm_std": ...,
        #     "history_len": ...
        #   }
        #
        # 旧形式:
        #   checkpoint 自体が state_dict
        # --------------------------------------------------
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            state = ckpt["model_state"]
            norm_mean = ckpt.get("norm_mean")
            norm_std = ckpt.get("norm_std")
            history_len = int(ckpt.get("history_len", self.history_len))
        else:
            state = ckpt
            norm_mean = None
            norm_std = None
            history_len = self.history_len

        self.history_len = history_len
        self._window = deque(maxlen=self.history_len)

        self._model = FishPolicy(history_len=self.history_len)
        self._model.load_state_dict(state)
        self._model.eval()

        self._norm_mean = np.array(norm_mean, dtype=np.float32) if norm_mean is not None else None
        self._norm_std = np.array(norm_std, dtype=np.float32) if norm_std is not None else None

        self._loaded = True
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled and self._loaded and self._model is not None

    def reset(self):
        self._window.clear()

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
        return len(self._window) >= self.min_history

    def predict_probability(self) -> Optional[float]:
        """現在の履歴から押す確率を返す"""
        if not self.enabled or not self.ready():
            return None

        if len(self._window) < self.history_len:
            pad_count = self.history_len - len(self._window)
            first = self._window[0]
            arr = [first.copy() for _ in range(pad_count)] + list(self._window)
        else:
            arr = list(self._window)

        x = np.concatenate(arr, axis=0).astype(np.float32)

        if self._norm_mean is not None and self._norm_std is not None:
            std = self._norm_std.copy()
            std[std < 1e-6] = 1.0
            x = (x - self._norm_mean) / std

        xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        prob = self._model.predict(xt)
        return float(prob)

    def decide(
        self,
        pd_press: bool,
        dist_ratio: float,
        error_px: float,
        probability: Optional[float] = None,
    ) -> HybridDecision:
        """
        pd_press:
            既存PDの判定結果
        dist_ratio:
            error / bar_h などの正規化誤差
        error_px:
            ピクセル単位の誤差
        probability:
            事前計算した押す確率（Noneなら内部計算）
        """
        if probability is None:
            probability = self.predict_probability()

        if probability is None:
            return HybridDecision(
                press=pd_press,
                probability=0.5,
                mode="fallback",
                confidence=0.0,
            )

        # 0.5 からの距離を confidence とみなす
        confidence = min(1.0, abs(probability - 0.5) * 2.0)

        # PDが迷いやすい領域のみ imitation に補助させる
        # 例: 魚がバー中心付近にいる場面
        in_assist_zone = abs(dist_ratio) <= self.assist_band

        # 強く押すべき / 強く離すべきと model が言っている時だけ
        # assist zone で PD を上書きする
        if in_assist_zone:
            if probability >= self.strong_press:
                return HybridDecision(
                    press=True,
                    probability=probability,
                    mode="assist_press",
                    confidence=confidence,
                )
            if probability <= self.strong_release:
                return HybridDecision(
                    press=False,
                    probability=probability,
                    mode="assist_release",
                    confidence=confidence,
                )

        # それ以外は PD を尊重
        return HybridDecision(
            press=pd_press,
            probability=probability,
            mode="pd",
            confidence=confidence,
        )