from __future__ import annotations

import argparse
import glob
import math
import os
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

import config
from imitation.model import FishPolicy

FEATURE_COLUMNS = [
    "error",
    "velocity",
    "bar_h",
    "fish_delta",
    "dist_ratio",
    "mouse_prev",
    "fish_in_bar",
    "press_streak",
    "predicted",
    "bar_accel",
]

LABEL_COLUMN = "pd_press"
OPTIONAL_WEIGHT_COLUMN = "green"


@dataclass
class WindowSample:
    x: np.ndarray
    y: float
    weight: float


class WindowDataset(Dataset):
    def __init__(self, samples: Sequence[WindowSample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return (
            torch.from_numpy(s.x).float(),
            torch.tensor(s.y, dtype=torch.float32),
            torch.tensor(s.weight, dtype=torch.float32),
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_csv_files(data_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"CSV が見つかりません: {data_dir}")
    return files


def load_one_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = FEATURE_COLUMNS + [LABEL_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(path)} に列が足りません: {missing}")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLUMNS + [LABEL_COLUMN]).copy()
    if df.empty:
        return df

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors="coerce")

    if OPTIONAL_WEIGHT_COLUMN in df.columns:
        df[OPTIONAL_WEIGHT_COLUMN] = pd.to_numeric(df[OPTIONAL_WEIGHT_COLUMN], errors="coerce").fillna(0.0)
    else:
        df[OPTIONAL_WEIGHT_COLUMN] = 0.0

    df = df.dropna(subset=FEATURE_COLUMNS + [LABEL_COLUMN]).copy()
    if df.empty:
        return df

    df[LABEL_COLUMN] = (df[LABEL_COLUMN].astype(float) > 0.5).astype(np.float32)
    return df


def split_files(files: Sequence[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    files = list(files)
    rng = random.Random(seed)
    rng.shuffle(files)

    if len(files) == 1:
        return files, files

    val_count = max(1, int(round(len(files) * val_ratio)))
    val_files = files[:val_count]
    train_files = files[val_count:] or files[:1]
    return train_files, val_files


def compute_norm_stats(files: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    feats_all = []
    for path in files:
        df = load_one_csv(path)
        if df.empty:
            continue
        feats_all.append(df[FEATURE_COLUMNS].to_numpy(dtype=np.float32))

    if not feats_all:
        raise RuntimeError("正規化統計を計算できません。CSV が空か壊れています。")

    x = np.concatenate(feats_all, axis=0)
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def build_samples_from_files(
    files: Sequence[str],
    history_len: int,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    min_weight: float,
    green_weight: float,
) -> List[WindowSample]:
    samples: List[WindowSample] = []

    for path in files:
        df = load_one_csv(path)
        if len(df) < history_len:
            continue

        feats = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        labels = df[LABEL_COLUMN].to_numpy(dtype=np.float32)
        greens = df[OPTIONAL_WEIGHT_COLUMN].to_numpy(dtype=np.float32)

        feats = (feats - norm_mean) / norm_std

        for end_idx in range(history_len - 1, len(df)):
            start_idx = end_idx - history_len + 1
            window = feats[start_idx:end_idx + 1].reshape(-1).astype(np.float32)
            y = float(labels[end_idx])
            g = float(np.clip(greens[end_idx], 0.0, 1.0))
            weight = float(max(min_weight, 1.0 + g * green_weight))
            samples.append(WindowSample(x=window, y=y, weight=weight))

    if not samples:
        raise RuntimeError("学習サンプルが 0 件です。history_len が長すぎるか、CSV が不足しています。")
    return samples


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    probs_all = []
    labels_all = []

    for x, y, w in loader:
        x = x.to(device)
        y = y.to(device)
        w = w.to(device)

        out = model(x).squeeze(-1)
        if out.min().item() < 0.0 or out.max().item() > 1.0:
            prob = torch.sigmoid(out)
        else:
            prob = out.clamp(1e-6, 1.0 - 1e-6)

        loss = F.binary_cross_entropy(prob, y, reduction="none")
        loss = (loss * w).mean()

        pred = (prob >= 0.5).float()
        total_loss += float(loss.item()) * x.size(0)
        total += x.size(0)
        correct += int((pred == y).sum().item())
        probs_all.append(prob.detach().cpu())
        labels_all.append(y.detach().cpu())

    probs = torch.cat(probs_all) if probs_all else torch.tensor([])
    labels = torch.cat(labels_all) if labels_all else torch.tensor([])

    pos_pred = int((probs >= 0.5).sum().item()) if len(probs) else 0
    pos_true = int((labels >= 0.5).sum().item()) if len(labels) else 0

    tp = int((((probs >= 0.5) & (labels >= 0.5))).sum().item()) if len(probs) else 0
    fp = int((((probs >= 0.5) & (labels < 0.5))).sum().item()) if len(probs) else 0
    fn = int((((probs < 0.5) & (labels >= 0.5))).sum().item()) if len(probs) else 0

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)

    return {
        "loss": total_loss / max(1, total),
        "acc": correct / max(1, total),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pos_pred": pos_pred,
        "pos_true": pos_true,
        "count": total,
    }


def save_checkpoint(
    save_path: str,
    model: nn.Module,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    history_len: int,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "norm_mean": torch.from_numpy(norm_mean.astype(np.float32)),
        "norm_std": torch.from_numpy(norm_std.astype(np.float32)),
        "history_len": history_len,
    }
    torch.save(payload, save_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="PD ログから imitation policy を学習")
    parser.add_argument("--data-dir", default=getattr(config, "PD_DATA_DIR", os.path.join(config.BASE_DIR, "data", "pd_sessions")), help="PD CSV ディレクトリ")
    parser.add_argument("--save-path", default=getattr(config, "IL_MODEL_PATH", os.path.join(config.BASE_DIR, "imitation", "policy.pt")), help="保存先 .pt")
    parser.add_argument("--history-len", type=int, default=getattr(config, "IL_HISTORY_LEN", 10), help="履歴フレーム数")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-weight", type=float, default=1.0, help="各サンプル最小重み")
    parser.add_argument("--green-weight", type=float, default=0.5, help="green 列に応じて加える重み")
    args = parser.parse_args()

    set_seed(args.seed)

    files = list_csv_files(args.data_dir)
    train_files, val_files = split_files(files, args.val_ratio, args.seed)

    print(f"[INFO] CSV total : {len(files)}")
    print(f"[INFO] train files: {len(train_files)}")
    print(f"[INFO] val files  : {len(val_files)}")

    norm_mean, norm_std = compute_norm_stats(train_files)

    train_samples = build_samples_from_files(
        train_files,
        history_len=args.history_len,
        norm_mean=norm_mean,
        norm_std=norm_std,
        min_weight=args.min_weight,
        green_weight=args.green_weight,
    )
    val_samples = build_samples_from_files(
        val_files,
        history_len=args.history_len,
        norm_mean=norm_mean,
        norm_std=norm_std,
        min_weight=args.min_weight,
        green_weight=args.green_weight,
    )

    print(f"[INFO] train samples: {len(train_samples)}")
    print(f"[INFO] val samples  : {len(val_samples)}")

    train_pos = sum(int(s.y >= 0.5) for s in train_samples)
    val_pos = sum(int(s.y >= 0.5) for s in val_samples)
    print(f"[INFO] train pos rate: {train_pos / max(1, len(train_samples)):.3f}")
    print(f"[INFO] val pos rate  : {val_pos / max(1, len(val_samples)):.3f}")

    train_loader = DataLoader(
        WindowDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        WindowDataset(val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FishPolicy(history_len=args.history_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = -1.0
    best_path = args.save_path

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for x, y, w in train_loader:
            x = x.to(device)
            y = y.to(device)
            w = w.to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(x).squeeze(-1)
            if out.min().item() < 0.0 or out.max().item() > 1.0:
                prob = torch.sigmoid(out)
            else:
                prob = out.clamp(1e-6, 1.0 - 1e-6)

            loss = F.binary_cross_entropy(prob, y, reduction="none")
            loss = (loss * w).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = x.size(0)
            running_loss += float(loss.item()) * bs
            seen += bs

        train_loss = running_loss / max(1, seen)
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"[E{epoch:03d}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['acc']:.4f} "
            f"f1={val_metrics['f1']:.4f} "
            f"prec={val_metrics['precision']:.4f} "
            f"recall={val_metrics['recall']:.4f} "
            f"pred_pos={val_metrics['pos_pred']} "
            f"true_pos={val_metrics['pos_true']}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            save_checkpoint(best_path, model, norm_mean, norm_std, args.history_len)
            print(f"[SAVE] best -> {best_path} (f1={best_f1:.4f})")

    print("[DONE] 学習完了")
    print(f"[DONE] best model: {best_path}")
    print("[NEXT] config.py の IL_MODEL_PATH をこの .pt に向けて、IL_USE_MODEL=True で試してください。")


if __name__ == "__main__":
    main()
