# train_adaptive_pd.py
from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config
from imitation.adaptive_pd import GainAdapterNet


LOG_COLUMNS = [
    "error",
    "velocity",
    "bar_h",
    "fish_delta",
    "dist_ratio",
    "mouse_pressed",
    "fish_in_bar",
    "press_streak",
    "predicted",
    "bar_accel",
]


@dataclass
class TrainResult:
    best_val_acc: float
    save_path: str
    train_samples: int
    val_samples: int


def default_log(msg: str):
    print(msg, flush=True)


def _get_model_path() -> str:
    return getattr(
        config,
        "ADAPTIVE_PD_MODEL_PATH",
        os.path.join(config.BASE_DIR, "imitation", "adaptive_pd.pt"),
    )


def _get_history_len() -> int:
    return int(getattr(config, "ADAPTIVE_PD_HISTORY_LEN", 10))


def _find_csvs() -> list[str]:
    data_dir = getattr(
        config,
        "IL_DATA_DIR",
        os.path.join(config.BASE_DIR, "imitation", "data"),
    )
    return sorted(glob.glob(os.path.join(data_dir, "session_*.csv")))


def _load_sessions(log_fn: Callable[[str], None]) -> list[pd.DataFrame]:
    csvs = _find_csvs()
    if not csvs:
        raise FileNotFoundError("session_*.csv が見つかりません")

    frames = []
    for path in csvs:
        try:
            df = pd.read_csv(path)
            missing = [c for c in LOG_COLUMNS if c not in df.columns]
            if missing:
                log_fn(f"[SKIP] {os.path.basename(path)} missing={missing}")
                continue
            df = df[LOG_COLUMNS].copy()
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            if len(df) < _get_history_len() + 2:
                log_fn(f"[SKIP] {os.path.basename(path)} too short: {len(df)}")
                continue
            frames.append(df)
        except Exception as e:
            log_fn(f"[SKIP] {os.path.basename(path)} read failed: {e}")

    if not frames:
        raise RuntimeError("学習に使える session CSV がありません")

    log_fn(f"[DATA] usable sessions: {len(frames)}")
    return frames


def _build_dataset(history_len: int, log_fn: Callable[[str], None]):
    sessions = _load_sessions(log_fn)

    xs = []
    ys = []

    for df in sessions:
        arr = df.to_numpy(dtype=np.float32)

        # 入力は 10次元のうち mouse_pressed 以外も含める。
        # ただし mouse_pressed は「前フレーム状態」として使うため、
        # 現フレームの教師ラベル漏れを避けたいなら、1フレームずらして使う。
        # ここでは簡単化のため、そのまま使う代わりに将来ラベルは使わない。
        for i in range(history_len, len(arr)):
            hist = arr[i - history_len:i].copy()

            # hist[:, 5] は mouse_pressed。
            # 最終フレームだけは未来ラベルに近くなるので、
            # 末尾フレームは 1つ前の値で上書きしてリークを弱める。
            hist[-1, 5] = hist[-2, 5]

            x = hist.reshape(-1)
            y = arr[i, 5]  # 現フレームの mouse_pressed

            xs.append(x)
            ys.append(y)

    X = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)

    if len(X) < 64:
        raise RuntimeError(f"サンプル不足: {len(X)}")

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    Xn = (X - mean) / std

    log_fn(f"[DATA] samples={len(Xn)} history_len={history_len}")
    return Xn, y, mean, std


class GainLoss(nn.Module):
    """
    ΔKp, ΔKd を出させて、
    既存 hold 式から mouse_pressed を再現するよう学習する。
    """

    def __init__(
        self,
        hold_gain: float,
        speed_damping: float,
        hold_min: float,
        hold_max: float,
        delta_kp_scale: float = 0.030,
        delta_kd_scale: float = 0.0025,
        logit_temp: float = 0.010,
        reg_lambda: float = 0.02,
    ):
        super().__init__()
        self.hold_gain = hold_gain
        self.speed_damping = speed_damping
        self.hold_min = hold_min
        self.hold_max = hold_max
        self.delta_kp_scale = delta_kp_scale
        self.delta_kd_scale = delta_kd_scale
        self.logit_temp = logit_temp
        self.reg_lambda = reg_lambda
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, out: torch.Tensor, x_flat: torch.Tensor, y_true: torch.Tensor):
        # x_flat shape = [B, history_len * 10]
        # 最新フレームの error / velocity を使う
        last = x_flat[:, -10:]
        error = last[:, 0]
        velocity = last[:, 1]

        delta_kp = out[:, 0] * self.delta_kp_scale
        delta_kd = out[:, 1] * self.delta_kd_scale

        kp = torch.clamp(self.hold_gain + delta_kp, min=0.0)
        kd = self.speed_damping + delta_kd

        hold = (
            self.hold_min
            + torch.abs(error) * kp
            + velocity * kd
        )
        hold = torch.clamp(hold, self.hold_min, self.hold_max)

        # hold が hold_min からどれだけ離れたかで press ロジット化
        logits = (hold - self.hold_min) / self.logit_temp

        cls_loss = self.bce(logits, y_true)
        reg_loss = (delta_kp.pow(2).mean() + delta_kd.pow(2).mean())
        loss = cls_loss + self.reg_lambda * reg_loss

        with torch.no_grad():
            pred = (torch.sigmoid(logits) > 0.5).float()
            acc = (pred == y_true).float().mean()

        return loss, cls_loss.detach(), reg_loss.detach(), acc.detach()


def run_training(
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_ratio: float = 0.2,
    log_fn: Callable[[str], None] = default_log,
) -> TrainResult:
    history_len = _get_history_len()
    X, y, mean, std = _build_dataset(history_len, log_fn)

    n = len(X)
    idx = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)

    n_val = max(1, int(n * val_ratio))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.float32)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32)
    y_val = torch.tensor(y[val_idx], dtype=torch.float32)

    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_fn(f"[TRAIN] device={device}")

    model = GainAdapterNet(history_len=history_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion = GainLoss(
        hold_gain=float(getattr(config, "HOLD_GAIN", 0.040)),
        speed_damping=float(getattr(config, "SPEED_DAMPING", 0.00025)),
        hold_min=float(getattr(config, "HOLD_MIN_S", 0.015)),
        hold_max=float(getattr(config, "HOLD_MAX_S", 0.100)),
    )

    X_val_d = X_val.to(device)
    y_val_d = y_val.to(device)

    best_val_acc = 0.0
    best_state = None

    log_fn(" Epoch | TrainLoss |  ValLoss | TrainAcc |  ValAcc | LR")
    log_fn("----------------------------------------------------------")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_count = 0

        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)

            out = model(xb)
            loss, _, _, acc = criterion(out, xb, yb)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = len(xb)
            total_loss += loss.item() * bs
            total_acc += acc.item() * bs
            total_count += bs

        train_loss = total_loss / max(1, total_count)
        train_acc = total_acc / max(1, total_count)

        model.eval()
        with torch.no_grad():
            out_val = model(X_val_d)
            val_loss_t, _, _, val_acc_t = criterion(out_val, X_val_d, y_val_d)
            val_loss = float(val_loss_t.item())
            val_acc = float(val_acc_t.item())

        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            log_fn(
                f"{epoch:>5} | "
                f"{train_loss:>9.4f} | "
                f"{val_loss:>8.4f} | "
                f"{train_acc:>8.1%} | "
                f"{val_acc:>7.1%} | "
                f"{lr_now:.6f}"
            )

    save_path = _get_model_path()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save(
        {
            "model_state": best_state,
            "norm_mean": mean,
            "norm_std": std,
            "history_len": history_len,
            "hold_gain": float(getattr(config, "HOLD_GAIN", 0.040)),
            "speed_damping": float(getattr(config, "SPEED_DAMPING", 0.00025)),
            "hold_min": float(getattr(config, "HOLD_MIN_S", 0.015)),
            "hold_max": float(getattr(config, "HOLD_MAX_S", 0.100)),
        },
        save_path,
    )

    log_fn("----------------------------------------------------------")
    log_fn(f"[DONE] best_val_acc={best_val_acc:.1%}")
    log_fn(f"[SAVE] {save_path}")

    return TrainResult(
        best_val_acc=best_val_acc,
        save_path=save_path,
        train_samples=len(train_idx),
        val_samples=len(val_idx),
    )


def main():
    parser = argparse.ArgumentParser(description="Adaptive PD trainer")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    run_training(
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        log_fn=default_log,
    )


if __name__ == "__main__":
    main()