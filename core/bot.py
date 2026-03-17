"""
釣りボットのメインロジック
========================

状態機械:
IDLE → CASTING → WAITING → HOOKING → FISHING → （ループ）

このクラスはバックグラウンドスレッドで動作するように設計されており、
共有プロパティを通じて GUI と通信する。
"""

import time
import cv2
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import config
from core.window import WindowManager
from core.screen import ScreenCapture
from core.detector import ImageDetector
from core.input_ctrl import InputController
from utils.logger import log

import ctypes
import csv
from collections import deque
import math
import random
import pickle


# 遅延ロード用のYOLO検出器
_yolo_detector = None
_yolo_device_used = None

class OnlineHoldRL:
    """
    高品質PDが出した base_hold に対して、
    delta_hold だけを学習する超軽量オンラインDouble Q学習器。
    replay buffer 対応。
    """

    def __init__(self):
        self.actions = list(getattr(
            config,
            "RL_HOLD_ACTIONS",
            [-0.05, -0.03, -0.015, -0.008, 0, 0.008, 0.015, 0.03, 0.05]
        ))
        self.alpha = float(getattr(config, "RL_ALPHA", 0.08))
        self.gamma = float(getattr(config, "RL_GAMMA", 0.96))
        self.epsilon = float(getattr(config, "RL_EPSILON", 0.25))

        # Double Q
        self.q1 = {}
        self.q2 = {}

        self.last_state = None
        self.last_action = None

        # replay buffer
        self.replay = deque(maxlen=int(getattr(config, "RL_REPLAY_SIZE", 5000)))
        self.replay_batch = int(getattr(config, "RL_REPLAY_BATCH", 8))
        self.replay_warmup = int(getattr(config, "RL_REPLAY_WARMUP", 200))

        self._load()

    def _key(self, state):
        return tuple(state)

    def _ensure_one(self, table, s):
        k = self._key(s)
        if k not in table:
            table[k] = [0.0 for _ in self.actions]
        return k

    def _ensure_both(self, s):
        k1 = self._ensure_one(self.q1, s)
        k2 = self._ensure_one(self.q2, s)
        return k1, k2

    def act(self, state):
        k1, k2 = self._ensure_both(state)

        # 行動選択は Q1+Q2 の合算で行う
        qv = [self.q1[k1][i] + self.q2[k2][i] for i in range(len(self.actions))]

        if random.random() < self.epsilon:
            a = random.randrange(len(self.actions))
        else:
            mx = max(qv)
            best = [i for i, v in enumerate(qv) if v == mx]
            a = random.choice(best)

        self.last_state = tuple(state)
        self.last_action = a
        return self.actions[a], a

    def _update_transition(self, state, action, reward, next_state, done=False):
        """
        Double Q-learning 更新
        50%でQ1を更新、50%でQ2を更新
        """
        s1, s2 = self._ensure_both(state)
        ns1, ns2 = self._ensure_both(next_state)

        if done:
            target = reward
            if random.random() < 0.5:
                q_old = self.q1[s1][action]
                self.q1[s1][action] = q_old + self.alpha * (target - q_old)
            else:
                q_old = self.q2[s2][action]
                self.q2[s2][action] = q_old + self.alpha * (target - q_old)
            return

        if random.random() < 0.5:
            # Q1 を更新: 行動選択はQ1、評価はQ2
            a_star = max(range(len(self.actions)), key=lambda i: self.q1[ns1][i])
            target = reward + self.gamma * self.q2[ns2][a_star]
            q_old = self.q1[s1][action]
            self.q1[s1][action] = q_old + self.alpha * (target - q_old)
        else:
            # Q2 を更新: 行動選択はQ2、評価はQ1
            a_star = max(range(len(self.actions)), key=lambda i: self.q2[ns2][i])
            target = reward + self.gamma * self.q1[ns1][a_star]
            q_old = self.q2[s2][action]
            self.q2[s2][action] = q_old + self.alpha * (target - q_old)

    def update(self, reward, next_state, done=False):
        if self.last_state is None or self.last_action is None:
            return

        state = self.last_state
        action = self.last_action
        next_state = tuple(next_state)

        # 今回の遷移をその場学習
        self._update_transition(state, action, reward, next_state, done)

        # replay buffer に保存
        self.replay.append((state, action, reward, next_state, done))

        # 過去経験も再学習
        if len(self.replay) >= self.replay_warmup:
            batch_n = min(self.replay_batch, len(self.replay))
            samples = random.sample(self.replay, batch_n)

            for s, a, r, ns, d in samples:
                self._update_transition(s, a, r, ns, d)

        if done:
            self.last_state = None
            self.last_action = None

    def save(self):
        path = getattr(config, "RL_MODEL_PATH", None)
        if not path:
            return

        try:
            with open(path, "wb") as f:
                pickle.dump({
                    "q1": self.q1,
                    "q2": self.q2,
                    "actions": self.actions,
                    "alpha": self.alpha,
                    "gamma": self.gamma,
                    "epsilon": self.epsilon,
                    "double_q": True,
                }, f)
        except Exception as e:
            log.warning(f"[RL] save失敗: {e}")

    def _load(self):
        path = getattr(config, "RL_MODEL_PATH", None)
        if not path or not os.path.exists(path):
            return

        try:
            with open(path, "rb") as f:
                d = pickle.load(f)

            loaded_actions = d.get("actions", None)
            if loaded_actions is not None and list(loaded_actions) != list(self.actions):
                log.warning(
                    "[RL] actions が現在設定と一致しないため、"
                    "既存Q-tableは読み込まず新規学習を開始します "
                    f"(saved={loaded_actions}, current={self.actions})"
                )
                self.q1 = {}
                self.q2 = {}
                return

            # 新形式: q1/q2
            if "q1" in d and "q2" in d:
                raw_q1 = d.get("q1", {})
                raw_q2 = d.get("q2", {})

                cleaned_q1 = {}
                cleaned_q2 = {}
                bad_count = 0

                all_keys = set(raw_q1.keys()) | set(raw_q2.keys())
                for k in all_keys:
                    v1 = raw_q1.get(k)
                    v2 = raw_q2.get(k)

                    ok1 = isinstance(v1, list) and len(v1) == len(self.actions)
                    ok2 = isinstance(v2, list) and len(v2) == len(self.actions)

                    if ok1 and ok2:
                        cleaned_q1[k] = v1
                        cleaned_q2[k] = v2
                    else:
                        bad_count += 1

                if bad_count > 0:
                    log.warning(f"[RL] action数不一致のstateを {bad_count} 件破棄しました")

                self.q1 = cleaned_q1
                self.q2 = cleaned_q2

                log.info(
                    f"[RL] Double Q-table 読み込み完了: {path} "
                    f"states={len(self.q1)}"
                )
                return

            # 旧形式: q だけ
            loaded_q = d.get("q", None)
            if loaded_q is not None:
                cleaned_q = {}
                bad_count = 0

                for k, v in loaded_q.items():
                    if isinstance(v, list) and len(v) == len(self.actions):
                        cleaned_q[k] = v
                    else:
                        bad_count += 1

                if bad_count > 0:
                    log.warning(f"[RL] action数不一致のstateを {bad_count} 件破棄しました")

                # 旧single QをDouble Qへ移行
                # Q1=旧Q, Q2=旧Qのコピーで開始
                self.q1 = {k: list(v) for k, v in cleaned_q.items()}
                self.q2 = {k: list(v) for k, v in cleaned_q.items()}

                log.info(
                    f"[RL] 旧single Q-table を Double Q に変換して読み込みました: "
                    f"{path} states={len(self.q1)}"
                )
                return

            # どちらも無い
            self.q1 = {}
            self.q2 = {}
            log.warning("[RL] 読み込めるQ-table形式が無かったため新規学習を開始します")

        except Exception as e:
            log.warning(f"[RL] load失敗: {e}")

def _get_yolo_detector(force_reload=False):
    """
    YOLO検出器を遅延ロードする。
    ultralytics が未インストールでも import 時点で落ちないようにするための工夫。

    Parameters
    ----------
    force_reload : bool
        True の場合は既存インスタンスを破棄して再生成する
    """
    global _yolo_detector, _yolo_device_used

    if force_reload:
        _yolo_detector = None

    if _yolo_detector is None or _yolo_device_used != config.YOLO_DEVICE:
        from core.yolo_detector import YoloDetector
        _yolo_detector = YoloDetector(config.YOLO_MODEL, conf=config.YOLO_CONF)
        _yolo_device_used = config.YOLO_DEVICE

    return _yolo_detector


class FishingBot:
    """VRChat用 自動釣りボット"""

    # 魚テンプレート名 → 表示名 + デバッグ枠色 (BGR)
    FISH_DISPLAY = {
        "fish_black":   ("黒魚",  (80, 80, 80)),
        "fish_white":   ("白魚",  (255, 255, 255)),
        "fish_copper":  ("銅魚",  (50, 127, 180)),
        "fish_green":   ("緑魚",  (0, 255, 0)),
        "fish_blue":    ("青魚",  (255, 150, 0)),
        "fish_purple":  ("紫魚",  (200, 50, 200)),
        "fish_golden":  ("金魚",  (0, 215, 255)),
        "fish_pink":    ("桃魚",  (180, 105, 255)),
        "fish_red":     ("赤魚",  (0, 0, 255)),
        "fish_rainbow": ("虹魚",  (0, 255, 255)),
    }

    def __init__(self):
        # 基本モジュール
        self.window   = WindowManager(config.WINDOW_TITLE)
        self.screen   = ScreenCapture()
        self.detector = ImageDetector(config.IMG_DIR, config.TEMPLATE_FILES)
        self.input    = InputController(self.window)
        

        # YOLO検出器
        self.yolo = None
        if config.USE_YOLO:
            try:
                self.yolo = _get_yolo_detector()
            except Exception as e:
                log.warning(f"[YOLO] 起動時ロード失敗: {e}")

        # ── GUI が読む共有状態 ──
        self.running    = False
        self.debug_mode = False
        self.fish_count = 0
        self.success_count = 0       # 釣り成功回数
        self.fail_count = 0          # 釣り失敗回数
        self.state      = "準備完了"

        # ── PD制御状態 ──
        self._bar_prev_cy   = None       # 前フレームの白バー中心Y
        self._bar_prev_time = None       # 前フレーム時刻
        self._bar_velocity  = 0.0        # 白バー速度推定 (px/s, 正=下, 負=上)
        self._last_hold     = None       # 前フレームの押下時間（後でフォールバックに使う）
        self._last_fish_cy  = None       # 前回の魚中心Y（後でフォールバックに使う）

        # ── PD教師データ収集 ──
        self._pd_writer = None
        self._pd_file = None
        self._pd_episode_id = 0
        self._pd_frame_idx = 0

        self._pd_prev_fish_cy = None
        self._pd_prev_bar_cy = None
        self._pd_prev_bar_v = 0.0
        self._pd_prev_mouse = 0
        self._pd_press_streak = 0

        self._pd_last_fish_box = None
        self._pd_last_bar_box = None
        self._pd_last_progress = 0.0
        self._pd_last_end_reason = ""

        # ── Residual RL ──
        self._rl = OnlineHoldRL() if getattr(config, "RL_ENABLE", False) else None
        self._rl_prev_state = None
        self._rl_prev_progress = 0.0
        self._rl_prev_abs_error = None
        self._rl_prev_delta_hold = 0.0
        self._rl_episode_reward = 0.0
        self._rl_step_reward_sum = 0.0      # 途中報酬の合計
        self._rl_terminal_reward = 0.0      # 最終報酬
        self._rl_prev_in_bar = True

        # ── Debug overlay（別スレッド描画。釣り処理を止めない） ──
        self._last_overlay_time = 0
        self._fps = 0.0
        self._frame_times = []
        self._debug_frame = None         # 最新の表示待ちフレーム
        self._debug_lock = threading.Lock()
        self._debug_thread = None

        # ── 回転補正状態 ──
        self._track_angle   = 0.0        # 軌道の傾き角度（度）
        self._need_rotation = False      # 回転補正が必要か

        # ── 強制リセット統計 ──
        self._retry_no_minigame_count = 0   # 連続でミニゲーム未検出の回数
        self._force_reset_count = 0         # 今回セッションの強制リセット総回数
        self._force_reset_log = []          # 強制リセットログ [{timestamp, count}, ...]
        # ★ ファイルには保存せず、今の起動中だけメモリ保持する

        # ── 魚 / 白バー位置平滑化（検出のブレ低減） ──
        self._fish_smooth_cy = None      # 平滑化した魚中心Y
        self._current_fish_name = ""     # 現在検出中の魚テンプレート名（例: fish_blue）
        self._bar_locked_cx  = None      # ★ 軌道X軸ロック（白バーと魚で共有）
        self._pool = ThreadPoolExecutor(max_workers=2)

        # ── 未検出ベースの頭向き補正状態 ──
        self._no_minigame_streak = 0
        self._no_minigame_adjust_active = False
        self._head_adjust_accum_sec = 0.0
        self._consecutive_fail = 0

        # ── 行動クローニング（Imitation Learning） ──
        self._il_history = deque(maxlen=config.IL_HISTORY_LEN)
        self._il_writer = None       # CSV writer（録画モード）
        self._il_file = None         # CSV file handle
        self._il_prev_fish_cy = None # 前フレーム魚Y（魚移動量計算用）
        self._il_mouse_prev = 0      # 前フレームのマウス状態
        self._il_log_counter = 0     # ログの出しすぎ防止カウンタ
        self._il_policy = None       # 学習済みモデル
        self._il_device = "cpu"
        self._il_norm_mean = None    # 特徴量正規化平均
        self._il_norm_std = None     # 特徴量正規化標準偏差

        if config.IL_USE_MODEL:
            self._load_il_policy()

    # ══════════════════════════════════════════════════════
    #  ゲーム画面取得
    # ══════════════════════════════════════════════════════

    def _grab(self):
        """
        VRChatウィンドウのクライアント領域をキャプチャする。
        必ず空でない BGR画像を返す。
        """
        try:
            img, _ = self.screen.grab_window(self.window)
            if img is not None and img.size > 0:
                return img
        except Exception:
            pass

        # 取得失敗時も後段でクラッシュしないようダミー画像を返す
        import numpy as np
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def _grab_rotated(self):
        """
        ウィンドウのクライアント領域を取得し、
        軌道に傾きがある場合は回転補正してから返す。
        """
        img = self._grab()
        if self._need_rotation:
            return self._rotate_for_detection(img)
        return img

    def _calculate_track_angle(self, track_box):
        """
        釣り軌道の傾き角度を計算する。

        Parameters
        ----------
        track_box : tuple
            軌道の境界ボックス (x, y, w, h)

        Returns
        -------
        float
            軌道傾き角度（度）
            正の値 = 右に傾いている
            負の値 = 左に傾いている
        """
        import numpy as np

        if track_box is None or len(track_box) < 4:
            return 0.0

        x, y, w, h = track_box[:4]

        # 軌道は本来かなり縦長のはずなので、その場合だけ角度計算する
        if h > w * 2:
            # 最小外接矩形を使って角度を推定
            # ここでは簡易的に4隅の点を使っている
            pts = np.array([
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h]
            ], dtype=np.float32)

            try:
                (cx, cy), (bw, bh), angle = cv2.minAreaRect(pts)

                # minAreaRect の角度は -90〜0 の独特な表現なので補正する
                if bw < bh:
                    angle = angle + 90

                # 大きすぎる角度は誤検出とみなす
                if abs(angle) > 45:
                    return 0.0

                return angle

            except Exception:
                return 0.0

        return 0.0

    def _record_force_reset(self):
        """強制リセットを1回記録する"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._force_reset_count += 1
        self._force_reset_log.append({
            "timestamp": timestamp,
            "count": self._force_reset_count
        })
        # ★ ファイル保存せずメモリだけ保持
        return timestamp

    def get_force_reset_log(self):
        """強制リセットログを取得する（GUI向け）"""
        return self._force_reset_log.copy()

    def clear_force_reset_log(self):
        """強制リセットログをクリアする"""
        self._force_reset_log = []
        self._force_reset_count = 0
        # ★ ファイル操作はしない

    def _rotate_for_detection(self, screen):
        """
        傾いた釣り軌道が垂直になるように画像を回転する。

        原理:
            軌道が θ° 傾いている
            → 画像を -θ° 回転
            → 軌道が垂直になる

        これにより、既存のテンプレートマッチング処理がそのまま使える。
        """
        import numpy as np

        h, w = screen.shape[:2]
        center = (w / 2.0, h / 2.0)

        # getRotationMatrix2D:
        # 画像座標系では正角度は時計回り回転
        # 軌道が右に θ° 傾く → 反時計回りに θ° 回したい → -θ を渡す
        M = cv2.getRotationMatrix2D(center, -self._track_angle, 1.0)

        # 回転後に画像が切れないようキャンバス拡大
        cos_a = abs(M[0, 0])
        sin_a = abs(M[0, 1])
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        return cv2.warpAffine(
            screen, M, (new_w, new_h), borderValue=(0, 0, 0)
        )

    # ══════════════════════════════════════════════════════
    #  PD教師データ収集
    # ══════════════════════════════════════════════════════

    def _pd_start_recording(self):
        """PD教師データ保存用CSVを開く（1セッション1ファイル）"""
        if self._pd_writer is not None:
            return

        os.makedirs(config.PD_DATA_DIR, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(config.PD_DATA_DIR, f"pd_session_{ts}.csv")

        self._pd_file = open(path, "w", newline="", encoding="utf-8")
        self._pd_writer = csv.writer(self._pd_file)

        self._pd_writer.writerow([
            "episode_id",
            "frame_idx",
            "t",
            "fish_name",

            # 生値
            "fish_cy",
            "bar_cy",
            "bar_h",

            # IL互換10特徴
            "error",
            "velocity",
            "fish_delta",
            "dist_ratio",
            "mouse",
            "fish_in_bar",
            "press_streak",
            "predicted",
            "bar_accel",

            # 教師ラベル
            "action_press",
            "control_value",
            "in_deadzone",

            # 解析用
            "progress",
            "end_reason",
            "episode_done",
            "episode_success",
        ])
        self._pd_file.flush()
        log.info(f"[PD_RECORD] 記録開始: {path}")

    def _pd_close_recording(self):
        """必要に応じてCSVを明示的に閉じる"""
        if self._pd_file is not None:
            try:
                self._pd_file.flush()
                self._pd_file.close()
            except Exception:
                pass

        self._pd_file = None
        self._pd_writer = None

    def _pd_reset_episode(self):
        """ミニゲーム1回分の状態をリセットする"""
        self._pd_episode_id += 1
        self._pd_frame_idx = 0
        self._pd_prev_fish_cy = None
        self._pd_prev_bar_cy = None
        self._pd_prev_bar_v = 0.0
        self._pd_prev_mouse = 0
        self._pd_press_streak = 0
        self._pd_last_fish_box = None
        self._pd_last_bar_box = None
        self._pd_last_progress = 0.0
        self._pd_last_end_reason = ""

    @staticmethod
    def _pd_clamp(v, lo, hi):
        return max(lo, min(hi, v))

    @staticmethod
    def _pd_safe_div(a, b, default=0.0):
        return a / b if abs(b) > 1e-6 else default

    def _pd_extract_features(self, fish_box, bar_box):
        """
        imitation/model.py と揃えた 10特徴量を生成する
        """
        fish_cy = float(fish_box[1] + fish_box[3] / 2.0)
        bar_cy = float(bar_box[1] + bar_box[3] / 2.0)
        bar_h = float(bar_box[3])

        error = bar_cy - fish_cy

        if self._pd_prev_bar_cy is None:
            velocity = 0.0
        else:
            velocity = bar_cy - self._pd_prev_bar_cy

        if self._pd_prev_fish_cy is None:
            fish_delta = 0.0
        else:
            fish_delta = fish_cy - self._pd_prev_fish_cy

        dist_ratio = self._pd_safe_div(error, max(bar_h, 1.0), 0.0)

        fish_in_bar = self._pd_safe_div(
            fish_cy - (bar_cy - bar_h / 2.0),
            max(bar_h, 1.0),
            0.5,
        )
        fish_in_bar = self._pd_clamp(fish_in_bar, -1.5, 2.5)

        press_streak_norm = self._pd_clamp(self._pd_press_streak / 30.0, -1.0, 1.0)
        predicted = error + velocity * 0.15
        bar_accel = velocity - self._pd_prev_bar_v

        return {
            "fish_cy": fish_cy,
            "bar_cy": bar_cy,
            "bar_h": bar_h,
            "error": float(error),
            "velocity": float(velocity),
            "fish_delta": float(fish_delta),
            "dist_ratio": float(dist_ratio),
            "mouse": float(self._pd_prev_mouse),
            "fish_in_bar": float(fish_in_bar),
            "press_streak": float(press_streak_norm),
            "predicted": float(predicted),
            "bar_accel": float(bar_accel),
        }

    def _pd_record_frame(
        self,
        *,
        fish_box,
        bar_box,
        fish_name,
        action_press,
        control_value,
        in_deadzone,
        progress=0.0,
        end_reason="",
        episode_done=0,
        episode_success=0,
    ):
        """1フレーム分のPD教師データを書き込む"""
        if not config.PD_RECORD:
            return
        if self._pd_writer is None:
            return
        if fish_box is None or bar_box is None:
            return

        feat = self._pd_extract_features(fish_box, bar_box)

        self._pd_writer.writerow([
            self._pd_episode_id,
            self._pd_frame_idx,
            time.time(),
            fish_name or "",

            feat["fish_cy"],
            feat["bar_cy"],
            feat["bar_h"],

            feat["error"],
            feat["velocity"],
            feat["fish_delta"],
            feat["dist_ratio"],
            feat["mouse"],
            feat["fish_in_bar"],
            feat["press_streak"],
            feat["predicted"],
            feat["bar_accel"],

            int(action_press),
            float(control_value),
            int(bool(in_deadzone)),

            float(progress),
            end_reason,
            int(episode_done),
            int(episode_success),
        ])

        if self._pd_frame_idx % 100 == 0:
            self._pd_file.flush()

        # 次フレーム用状態更新
        if int(action_press) == self._pd_prev_mouse:
            if int(action_press) == 1:
                self._pd_press_streak += 1
            else:
                self._pd_press_streak -= 1
        else:
            self._pd_press_streak = 1 if int(action_press) == 1 else -1

        self._pd_prev_mouse = int(action_press)
        self._pd_prev_fish_cy = feat["fish_cy"]
        self._pd_prev_bar_cy = feat["bar_cy"]
        self._pd_prev_bar_v = feat["velocity"]
        self._pd_frame_idx += 1

    def _pd_record_episode_end(self, *, fish_box, bar_box, fish_name, progress, success, end_reason):
        """
        最終1行を done=1 として追加する。
        最後の fish/bar が無い場合は書き込まない。
        """
        if fish_box is None or bar_box is None:
            return

        self._pd_record_frame(
            fish_box=fish_box,
            bar_box=bar_box,
            fish_name=fish_name,
            action_press=self._pd_prev_mouse,
            control_value=0.0,
            in_deadzone=False,
            progress=progress,
            end_reason=end_reason,
            episode_done=1,
            episode_success=1 if success else 0,
        )

    # ══════════════════════════════════════════════════════
    #  第1段階: 竿を投げる
    # ══════════════════════════════════════════════════════

    def _cast_rod(self):
        """竿を投げる"""
        self.state = "投竿中"

        if config.IL_RECORD:
            log.info("[🎣 投竿] 録画モード — 手動で竿を投げてください（マウスクリック）")
        else:
            shake_th = getattr(config, "SHAKE_HEAD_FAIL_THRESHOLD", 5)

            if self._consecutive_fail >= shake_th:
                log.warning(
                    f"[🎯 視点補正] 連続失敗 {self._consecutive_fail} 回 "
                    f"(閾値 {shake_th}) のため首振りします"
                )
                self.input.shake_head()
                time.sleep(0.15)
                self._consecutive_fail = 0

            log.info("[🎣 投竿] 投竿...")
            self.input.click()

        # ★ 投竿開始時点から debug ウィンドウを表示
        try:
            screen = self._grab()
            self._show_debug_overlay(screen, status_text="🎣 投竿中...")
        except Exception:
            pass

        time.sleep(config.CAST_DELAY)


    # ───────────────── 頭向き補正の更新 ──────────────────

    def _update_section_head_adjust(self, no_minigame: bool):
        if not getattr(config, "ENABLE_SECTION_HEAD_ADJUST", False):
            return

        step_sec = getattr(config, "HEAD_ADJUST_STEP_SEC", 0.3)
        detect_fail_th = getattr(config, "HEAD_ADJUST_FAIL_THRESHOLD", 10)

        if no_minigame:
            self._no_minigame_streak += 1
            log.info(
                f"[頭向き補正] 未検出カウント "
                f"{self._no_minigame_streak}/{detect_fail_th}"
            )

            if self._no_minigame_streak >= detect_fail_th:
                if not self._no_minigame_adjust_active:
                    self._no_minigame_adjust_active = True
                    log.warning(
                        f"[頭向き補正] 連続未検出 "
                        f"{self._no_minigame_streak}/{detect_fail_th} に到達。"
                        f"以後、未検出のたびに左補正します"
                    )

                log.info(f"[頭向き補正] 左へ {step_sec:.1f}s 補正を入れます")
                self.input.look_left_for(step_sec)
                self._head_adjust_accum_sec += step_sec

        else:
            if self._no_minigame_streak > 0 or self._no_minigame_adjust_active:
                log.info(
                    f"[頭向き補正] ミニゲーム検出を確認。"
                    f"未検出カウント {self._no_minigame_streak} をリセットします"
                )

            if self._head_adjust_accum_sec > 0:
                log.info(
                    f"[頭向き補正] 右へ {self._head_adjust_accum_sec:.2f}s 戻します"
                )
                self.input.look_right_for(self._head_adjust_accum_sec)
                self._head_adjust_accum_sec = 0.0

            self._no_minigame_streak = 0
            self._no_minigame_adjust_active = False

    # ══════════════════════════════════════════════════════
    #  第2段階: 食いつきを待つ
    # ══════════════════════════════════════════════════════

    def _wait_for_bite(self) -> bool:
        """魚が食いつくまで待つ"""
        self.state = "食いつき待ち"

        if config.IL_RECORD:
            wait_s = config.MINIGAME_TIMEOUT
            log.info(f"[⏳ 待機] 録画モード — 手動操作してください。ミニゲーム出現待ち（最大{wait_s:.0f}s）...")
        else:
            wait_s = config.BITE_FORCE_HOOK
            log.info(f"[⏳ 待機] {wait_s:.0f}s 後に自動で合わせます...")

        t0 = time.time()

        while self.running:
            elapsed = time.time() - t0

            if elapsed >= wait_s:
                log.info(f"[🪝 合わせ] {elapsed:.1f}s 待機完了、自動で合わせます")
                return True

            # debug ウィンドウ更新
            try:
                screen = self._grab()
                self._show_debug_overlay(
                    screen,
                    status_text=f"⏳ 合わせ待ち ({elapsed:.0f}/{wait_s:.0f}s)"
                )
            except Exception:
                pass

            time.sleep(0.2)

        return False

    # ══════════════════════════════════════════════════════
    #  第3段階: 合わせる
    # ══════════════════════════════════════════════════════

    def _hook_fish(self):
        """魚を合わせる（クリックしてミニゲームへ入る）"""
        self.state = "合わせ"

        if config.IL_RECORD:
            log.info("[🪝 合わせ] 録画モード — 手動で合わせてください（マウスクリック）")
        else:
            log.info("[🪝 合わせ] マウスクリックで合わせます！")
            time.sleep(config.HOOK_PRE_DELAY)
            self.input.click()

        # ★ 合わせ後しばらく debug を更新しながら UI出現待ち
        t0 = time.time()
        while time.time() - t0 < config.HOOK_POST_DELAY:
            try:
                screen = self._grab()
                self._show_debug_overlay(
                    screen, status_text="🪝 合わせ！ ミニゲームUI待機中..."
                )
            except Exception:
                pass
            time.sleep(0.05)

    def _verify_minigame(self) -> bool:
        """
        合わせた後、本当に釣りミニゲームUIが出現したか確認する。

        厳密な連続Nフレームではなく、
        **累積** Nフレーム検出で確認する。

        優先順位:
        1. YOLO（使える場合）
        2. テンプレートマッチング（フォールバック）
        """
        self.state = "ミニゲーム確認"
        log.info("[🔍 確認] ミニゲームUIを高速チェック中...")

        t0 = time.time()
        hit_count = 0
        required = config.VERIFY_CONSECUTIVE
        _use_yolo = config.USE_YOLO and self.yolo is not None

        # 回転補正状態リセット
        self._track_angle = 0.0
        self._need_rotation = False
        detected_angle = None

        while self.running and (time.time() - t0 < config.VERIFY_TIMEOUT):
            screen = self._grab()
            found = False

            self._show_debug_overlay(
                screen,
                status_text=f"🔍 UI確認 ({hit_count}/{required})"
            )

            _roi = config.DETECT_ROI

            # ── まず YOLO を優先 ──
            if _use_yolo:
                try:
                    det = self.yolo.detect(screen, _roi)

                    if det.get("bar") and det.get("track"):
                        yb = det["bar"]
                        yt = det["track"]

                        bar_cx = yb[0] + yb[2] // 2
                        track_cx = yt[0] + yt[2] // 2

                        # 軌道と白バーのX位置が近ければ有効とみなす
                        if abs(bar_cx - track_cx) < 150:
                            found = True

                            # ★ 軌道傾き計算
                            detected_angle = self._calculate_track_angle(yt)
                except Exception:
                    pass

            # ── テンプレートマッチングでフォールバック ──
            if not found:
                bar = self.detector.find_multiscale(
                    screen, "bar", config.THRESH_BAR,
                    scales=config.BAR_SCALES,
                    search_region=_roi,
                )
                track = self.detector.find_multiscale(
                    screen, "track", config.THRESH_TRACK,
                    search_region=_roi,
                )

                bar_cx = (bar[0] + bar[2] // 2) if bar else None
                track_cx = (track[0] + track[2] // 2) if track else None

                if bar_cx is not None and track_cx is not None:
                    if abs(bar_cx - track_cx) < 150:
                        found = True
                        detected_angle = self._calculate_track_angle(track)

            if found:
                hit_count += 1

                if hit_count >= required:
                    if detected_angle is not None:
                        self._track_angle = detected_angle
                        angle_abs = abs(self._track_angle)

                        # 一定角度以上なら回転補正を有効にする
                        self._need_rotation = (
                            angle_abs > config.TRACK_MIN_ANGLE
                            and angle_abs <= config.TRACK_MAX_ANGLE
                        )

                    log.info(
                        f"[✓ 確認] UIを検出！ "
                        f"(所要 {time.time()-t0:.1f}s"
                        f", angle={self._track_angle:.1f}°)"
                    )
                    return True

            time.sleep(0.03)

        log.warning(
            f"[✗ 誤動作] {config.VERIFY_TIMEOUT:.1f}s 以内にミニゲームUI確認できず "
            f"(累積ヒット: {hit_count}/{required})。投竿からやり直します"
        )
        return False
    def _wait_for_minigame_ui(self) -> bool:
        """
        録画モード専用:
        ミニゲームUIが出るまで待機する。

        条件:
        - 白バーと軌道の両方を検出
        - さらに連続3フレーム確認
        誤検出で入らないようにしている。
        """
        consecutive = 0
        required = 3
        _roi = config.DETECT_ROI
        logged = False

        while self.running:
            screen = self._grab()
            self._show_debug_overlay(
                screen,
                status_text=f"[IL] ミニゲーム待機中... ({consecutive}/{required})"
            )

            bar = self.detector.find_multiscale(
                screen, "bar", config.THRESH_BAR,
                scales=config.BAR_SCALES, search_region=_roi,
            )
            track = self.detector.find_multiscale(
                screen, "track", config.THRESH_TRACK,
                search_region=_roi,
            )

            if bar is not None and track is not None:
                bar_cx = bar[0] + bar[2] // 2
                track_cx = track[0] + track[2] // 2

                if abs(bar_cx - track_cx) < 150:
                    consecutive += 1

                    if not logged and consecutive >= 1:
                        log.info(f"[IL] UI要素検出 ({consecutive}/{required})...")
                        logged = True

                    if consecutive >= required:
                        log.info(
                            f"[IL] ミニゲーム確認完了! "
                            f"(連続{required}フレームで白バー+軌道を検出)"
                        )
                        return True
                else:
                    consecutive = 0
                    logged = False
            else:
                consecutive = 0
                logged = False

            time.sleep(0.1)

        return False

    # ══════════════════════════════════════════════════════
    #  第4段階: 釣りミニゲーム
    # ══════════════════════════════════════════════════════

    def _fishing_minigame(self) -> bool:
        """
        釣りミニゲーム本体。

        役割:
        - 魚 / 白バー / 進捗 の検出
        - PD or IL による入力制御
        - 成功 / 失敗判定
        """
        self.state = "ミニゲーム中"
        log.info("[🐟 釣り] ミニゲーム開始")

        # ── 行動クローニング状態を毎回リセット ──
        self._il_history.clear()
        self._il_prev_fish_cy = None
        self._il_mouse_prev = 0
        self._il_press_streak = 0
        self._il_prev_velocity = 0.0
        self._il_log_counter = 0

        self._rl_prev_state = None
        self._rl_prev_progress = 0.0
        self._rl_prev_abs_error = None
        self._rl_prev_delta_hold = 0.0
        self._rl_episode_reward = 0.0
        self._rl_step_reward_sum = 0.0
        self._rl_terminal_reward = 0.0
        self._rl_prev_in_bar = True

        # ── PD教師データ収集状態を毎回リセット ──
        if config.PD_RECORD and (not config.IL_RECORD) and (not config.IL_USE_MODEL):
            self._pd_start_recording()
            self._pd_reset_episode()
            log.info("[PD_RECORD] 今回のミニゲームを教師データとして記録します")

        # ── 制御モード表示 ──
        if config.IL_RECORD:
            self._il_start_recording()
            log.info("[IL] 録画モード: 手動で白バーを操作してください")
        elif config.IL_USE_MODEL:
            if self._il_policy is None:
                self._load_il_policy()

            if self._il_policy is not None:
                log.info("[IL] ★ 今回は行動クローニングモデルで制御します ★")
            else:
                log.warning("[IL] モデル読み込み失敗のため、PD制御へフォールバックします")
        else:
            log.info("[PD] 今回はPD制御を使用します")

        # ── YOLOモード（必要時に遅延ロード） ──
        if config.USE_YOLO and self.yolo is None:
            try:
                self.yolo = _get_yolo_detector()
            except Exception as e:
                log.warning(f"[YOLO] 読み込み失敗: {e}。テンプレートマッチングへフォールバックします")

        _use_yolo = config.USE_YOLO and self.yolo is not None
        if _use_yolo:
            log.info("[YOLO] YOLO物体検出を使用します")

        # 最初だけ詳細デバッグを有効化
        self.detector.debug_report = True

        # PostMessageモードでは前面化不要、クリック座標だけ更新
        self.input.move_to_game_center()

        # ── ミニゲーム内部状態 ──
        no_detect = 0
        fish_lost = 0              # 連続で魚を見失ったフレーム数
        frame = 0
        hold_count = 0             # 押下回数
        success = False
        _skip_fish = False         # ホワイトリスト外魚なら放棄
        _fish_id_saved = False     # 魚種識別デバッグ画像は1回だけ保存
        self._progress_debug_saved = False
        minigame_start = time.time()
        ui_gone_count = 0          # UI消失カウンタ
        had_good_detection = False # 一度でも魚+バーを正常検出したか
        track_alive = True         # 軌道が存在しているか
        obj_gone_count = 0         # 検出対象不足の連続フレーム数
        fish_gone_since = None     # 魚を見失い始めた時刻
        bar_gone_since = None      # 白バーを見失い始めた時刻

        # ── PD制御状態リセット ──
        self._bar_prev_cy = None
        self._bar_prev_time = None
        self._bar_velocity = 0.0
        self._last_hold = None
        self._last_fish_cy = None
        self._fish_smooth_cy = None
        self._bar_locked_cx = None

        # ── テンプレートロック用（後続フレーム高速化） ──
        locked_fish_key = None
        locked_fish_scales = None
        locked_bar_scales = None
        _BAR_X_HALF = config.REGION_X
        _FISH_X_HALF = max(config.REGION_X * 2, 80)

        # 初期画面取得
        screen_orig = self._grab()

        # ミニゲーム開始時の原寸スクリーンショット保存
        self.screen.save_debug(screen_orig, "minigame_start")
        h_orig, w_orig = screen_orig.shape[:2]
        log.info(f"  スクリーンショットサイズ: {w_orig}×{h_orig}")

        # 初期化段階でも debug overlay を表示
        self._show_debug_overlay(
            screen_orig, status_text="🐟 ミニゲーム初期化中..."
        )

        # 必要なら回転補正
        if self._need_rotation:
            log.info(
                f"  ► 軌道が {self._track_angle:.1f}° 傾いているため、"
                f"回転補正を有効化します（{-self._track_angle:.1f}° 回転）"
            )
            screen = self._rotate_for_detection(screen_orig)
        else:
            screen = screen_orig

        h_scr, w_scr = screen.shape[:2]

        # ── 検索範囲初期化 ──
        if _use_yolo:
            search_region = None
            bar_search_region = None
            _regions_locked = True

            if config.DETECT_ROI:
                log.info(
                    f"  [YOLO] ROI使用: "
                    f"X={config.DETECT_ROI[0]} Y={config.DETECT_ROI[1]} "
                    f"{config.DETECT_ROI[2]}x{config.DETECT_ROI[3]}"
                )
            else:
                log.info("  [YOLO] 全画面検出")
        else:
            search_region, track_cx, bar_search_region = self._init_search_region(screen)
            _regions_locked = False

            if track_cx is not None:
                self._bar_locked_cx = track_cx
                log.info(f"  ★ 軌道X軸を事前ロック: X={track_cx}")

            if search_region:
                srx, sry, srw, srh = search_region
                log.info(
                    f"  初期魚探索範囲: X={srx}~{srx+srw} Y={sry}~{sry+srh}"
                )

            if bar_search_region:
                bsx, bsy, bsw, bsh = bar_search_region
                log.info(
                    f"  初期白バー探索範囲: X={bsx}~{bsx+bsw} "
                    f"Y={bsy}~{bsy+bsh}（下半分）"
                )

        # ── 開幕安定押し ──
        if config.IL_RECORD:
            log.info("  ► 録画モードなので開幕押下はスキップ。手動操作してください")
        else:
            press_t = getattr(config, 'INITIAL_PRESS_TIME', 0.2)
            log.info(f"  ► 開幕遅延 0.5秒 + 押下 {press_t}s")
            time.sleep(0.5)
            self.input.mouse_down()
            time.sleep(press_t)
            self.input.mouse_up()

        _last_progress_sr = None
        _last_track_w = None
        _last_green = 0.0
        _PROGRESS_SKIP_FRAMES = 0
        _prev_green = 0.0

        # ── 成功判定用: 終了直前60フレームの進捗履歴 ──
        green_history = deque(maxlen=20)

        # ── 直前10フレーム補完用バッファ ──
        recent_fish = deque(maxlen=10)
        recent_bar = deque(maxlen=10)
        recent_progress = deque(maxlen=10)
        recent_hook = deque(maxlen=10)

        def _push_recent(buf, value, frame_no):
            """補完用履歴バッファに追加する"""
            if value is not None:
                buf.append((frame_no, value))

        def _get_recent(buf, frame_no, max_age=5):
            """直近 max_age フレーム以内の値を取得する"""
            for fno, val in reversed(buf):
                if frame_no - fno <= max_age:
                    return val
            return None

        end_reason = "unknown"

        try:
            while self.running:
                frame += 1

                # ── FPS計算 ──
                now_t = time.time()
                self._frame_times.append(now_t)
                if len(self._frame_times) > 20:
                    self._frame_times = self._frame_times[-20:]

                if len(self._frame_times) >= 2:
                    dt = self._frame_times[-1] - self._frame_times[0]
                    if dt > 0:
                        self._fps = (len(self._frame_times) - 1) / dt

                # 生画面取得
                screen_raw = self._grab()
                screen = self._rotate_for_detection(screen_raw) \
                    if self._need_rotation else screen_raw

                # ════════════ タイムアウト判定 ════════════
                elapsed = time.time() - minigame_start
                if elapsed > config.MINIGAME_TIMEOUT:
                    log.info(
                        f"[⏱ タイムアウト] ミニゲームが {elapsed:.0f}s 続き、"
                        f"制限 {config.MINIGAME_TIMEOUT:.0f}s を超えたため終了します"
                    )
                    end_reason = "minigame_timeout"
                    break

                # ════════════ 定期的なUI存在確認 ════════════
                if frame % config.UI_CHECK_FRAMES == 0 and frame > 10:
                    if _use_yolo:
                        _tc = self.yolo.detect(screen, config.DETECT_ROI)
                        track_check = _tc["track"]
                    else:
                        track_check = self.detector.find_multiscale(
                            screen, "track", 0.50
                        )

                    if track_check is None:
                        ui_gone_count += 1
                        track_alive = False

                        log.info(
                            f"[⚠ UI確認] 軌道を検出できません "
                            f"({ui_gone_count}/{config.UI_GONE_LIMIT})"
                        )

                        if ui_gone_count >= config.UI_GONE_LIMIT:
                            log.info("[📋 終了] ミニゲームUIが消えたため終了します")
                            end_reason = "ui_gone_limit"
                            break
                    else:
                        ui_gone_count = 0
                        track_alive = True

                # 60フレームごとにカーソル位置の安全補正
                if frame % 60 == 0:
                    self.input.ensure_cursor_in_game()

                # ════════════ 連続ロスト時は軽量探索 ════════════
                if no_detect > 3 and not _use_yolo:
                    bar_quick = self.detector.find_multiscale(
                        screen, "bar", config.THRESH_BAR,
                        bar_search_region,
                        scales=locked_bar_scales or config.BAR_SCALES,
                    )

                    if bar_quick is not None:
                        log.info(f"[✓ 復帰] {no_detect}フレーム見失い後に白バー再検出")
                        no_detect = 0
                    else:
                        no_detect += 1

                        if no_detect > 5:
                            self.input.mouse_up()

                        if no_detect > config.TRACK_LOST_LIMIT:
                            log.info(
                                f"[📋 終了] {no_detect}フレーム連続で"
                                f"有効UIを検出できなかったため終了します"
                            )
                            end_reason = "track_lost_limit_quick"
                            break

                        self._show_debug_overlay(
                            screen_raw,
                            status_text=f"⚠ 見失い中 {no_detect}/{config.TRACK_LOST_LIMIT}"
                        )
                        time.sleep(config.GAME_LOOP_INTERVAL)
                        continue

                # ════════════ 魚 + 白バー検出 ════════════
                fish = None
                bar = None
                fish_detect_name = ""
                _matched_key = None
                _bar_scale = 1.0
                _yolo_progress = None
                _yolo_hook = None

                if _use_yolo:
                    _yolo_roi = config.DETECT_ROI
                    _ydet = self.yolo.detect(screen, roi=_yolo_roi)

                    fish = _ydet["fish"]
                    bar = _ydet["bar"]
                    _yolo_progress = _ydet.get("progress")
                    _yolo_hook = _ydet.get("hook")

                    # YOLOで魚の位置を取ったあと、色で魚種を判別
                    if fish is not None:
                        _save = not _fish_id_saved
                        _color_key = self.detector.identify_fish_type(
                            screen, fish, debug_save=_save
                        )

                        if _save:
                            _fish_id_saved = True

                        _matched_key = _color_key
                        fish_detect_name = _color_key
                    else:
                        _matched_key = None
                        fish_detect_name = ""

                    # YOLO用データ収集（失敗時専用でない場合）
                    _now = time.time()
                    if config.YOLO_COLLECT and not config.YOLO_COLLECT_ON_FAIL:
                        if not hasattr(self, '_yolo_last_collect_time'):
                            self._yolo_last_collect_time = 0

                        if _now - self._yolo_last_collect_time >= 60:
                            self._yolo_last_collect_time = _now
                            _cdir = os.path.join(
                                config.BASE_DIR, "yolo", "dataset",
                                "images", "unlabeled"
                            )
                            os.makedirs(_cdir, exist_ok=True)
                            _ts = time.strftime("%Y%m%d_%H%M%S")
                            _ms = int((_now % 1) * 1000)

                            cv2.imwrite(
                                os.path.join(_cdir, f"{_ts}_{_ms:03d}.png"),
                                screen
                            )

                else:
                    # ── テンプレートマッチング経路 ──
                    _fish_sr = search_region

                    if search_region:
                        _sr_x, _sr_y, _sr_w, _sr_h = search_region
                        _new_x, _new_w = _sr_x, _sr_w
                        _new_y, _new_h = _sr_y, _sr_h

                        # X方向は軌道中央付近に絞る
                        if self._bar_locked_cx is not None:
                            _nx = max(_sr_x, self._bar_locked_cx - _FISH_X_HALF)
                            _nx2 = min(_sr_x + _sr_w, self._bar_locked_cx + _FISH_X_HALF)
                            if _nx2 - _nx > 10:
                                _new_x, _new_w = _nx, _nx2 - _nx

                        # Y方向は平滑化魚位置の近辺に絞る
                        if self._fish_smooth_cy is not None:
                            _ny = max(_sr_y, int(self._fish_smooth_cy) - 150)
                            _ny2 = min(_sr_y + _sr_h, int(self._fish_smooth_cy) + 150)
                            if _ny2 - _ny > 30:
                                _new_y, _new_h = _ny, _ny2 - _ny

                        _fish_sr = (_new_x, _new_y, _new_w, _new_h)

                    # グレースケールを事前計算して再利用
                    _fg, _fox, _foy = self.detector.prepare_gray(
                        screen, _fish_sr, upload_gpu=True
                    )
                    _bg, _box, _boy = self.detector.prepare_gray(
                        screen, bar_search_region, upload_gpu=True
                    )

                    _has_cuda = self.detector._use_cuda

                    def _detect_fish():
                        """魚検出（必要ならロック済みテンプレ優先）"""
                        if locked_fish_key:
                            r = self.detector.find_multiscale(
                                screen, locked_fish_key, config.THRESH_FISH,
                                _fish_sr, scales=locked_fish_scales,
                                pre_gray=_fg, pre_offset=(_fox, _foy),
                            )

                            if r is None and _fish_sr is not search_region:
                                r = self.detector.find_multiscale(
                                    screen, locked_fish_key,
                                    config.THRESH_FISH,
                                    search_region, scales=locked_fish_scales
                                )

                            return r, locked_fish_key if r else None

                        else:
                            if _has_cuda:
                                r = self.detector.find_fish(
                                    screen, config.THRESH_FISH, _fish_sr,
                                    pre_gray=_fg, pre_offset=(_fox, _foy),
                                )
                            else:
                                # CPU時は全魚テンプレを毎フレーム走らせず、グループ分割して回す
                                _n = len(config.FISH_KEYS)
                                _grp_size = 2
                                _grp_count = ((_n + _grp_size - 1) // _grp_size)
                                _grp_idx = frame % _grp_count
                                _start = _grp_idx * _grp_size
                                _keys = config.FISH_KEYS[_start:_start + _grp_size]

                                r = self.detector.find_fish(
                                    screen, config.THRESH_FISH, _fish_sr,
                                    pre_gray=_fg, pre_offset=(_fox, _foy),
                                    keys=_keys,
                                )

                            return (r, self.detector._last_best_key if r else None)

                    def _detect_bar():
                        """白バー検出"""
                        _scales = locked_bar_scales or config.BAR_SCALES
                        r = self.detector.find_multiscale(
                            screen, "bar", config.THRESH_BAR,
                            bar_search_region, scales=_scales,
                            pre_gray=_bg, pre_offset=(_box, _boy),
                        )
                        return r, self.detector._last_scale

                    # 魚と白バーを並列検出
                    fut_fish = self._pool.submit(_detect_fish)
                    fut_bar = self._pool.submit(_detect_bar)

                    fish_result = fut_fish.result()
                    bar_result = fut_bar.result()

                    fish, _matched_key = fish_result
                    bar, _bar_scale = bar_result

                # ── テンプレ経路時の魚種ロック ──
                if not _use_yolo:
                    fish_detect_name = ""

                    if locked_fish_key:
                        if fish is not None:
                            fish_detect_name = locked_fish_key

                        # 一定時間見失ったらロック解除
                        if (fish is None and fish_lost > 20
                                and fish_lost % 20 == 0):
                            locked_fish_key = None
                            locked_fish_scales = None
                            log.info("  ★ 魚テンプレートロック解除、再探索します")
                    else:
                        if fish is not None:
                            fish_detect_name = _matched_key or "?"

                            # 白魚以外は魚種ロックして高速化
                            if (_matched_key and _matched_key != "fish_white"):
                                locked_fish_key = _matched_key
                                s = self.detector._last_best_scale
                                locked_fish_scales = [
                                    round(s * 0.85, 2), s, round(s * 1.15, 2)
                                ]
                                log.info(
                                    f"  ★ 魚テンプレートをロック: "
                                    f"{locked_fish_key} @ scales="
                                    f"{[f'{x:.2f}' for x in locked_fish_scales]}"
                                )

                # ── 補完前の有効検出だけ履歴に保存 ──
                _push_recent(recent_fish, fish, frame)
                _push_recent(recent_bar, bar, frame)
                _push_recent(recent_progress, _yolo_progress, frame)
                _push_recent(recent_hook, _yolo_hook, frame)

                # 魚名更新 & ホワイトリスト判定
                if fish is not None:
                    self._current_fish_name = fish_detect_name

                    if not _skip_fish and fish_detect_name:
                        wl_key = fish_detect_name
                        if not config.FISH_WHITELIST.get(wl_key, True):
                            fname_cn = self.FISH_DISPLAY.get(wl_key, (wl_key,))[0]
                            log.info(f"[ホワイトリスト] {fname_cn} は対象外のため、この釣りは放棄します")
                            _skip_fish = True

                # 白バーのスケールもロックして高速化
                if not _use_yolo and bar is not None and not locked_bar_scales:
                    locked_bar_scales = [
                        round(max(0.2, _bar_scale * 0.85), 2),
                        _bar_scale,
                        round(_bar_scale * 1.15, 2),
                    ]
                    log.info(
                        f"  ★ 白バースケールをロック "
                        f"@ scales={[f'{x:.2f}' for x in locked_bar_scales]}"
                    )

                # ════════════ X軸妥当性確認（魚と白バーは同じ軌道Xを共有） ════════════
                if bar is not None:
                    raw_bcx = bar[0] + bar[2] // 2

                    if self._bar_locked_cx is None:
                        self._bar_locked_cx = raw_bcx
                        log.info(f"  ★ 軌道X軸をロック(白バー): X={raw_bcx}")
                    elif abs(raw_bcx - self._bar_locked_cx) > _BAR_X_HALF:
                        bar = None

                    if bar is not None:
                        # X軸揺れを消すためロック座標へ合わせる
                        bar = (
                            self._bar_locked_cx - bar[2] // 2,
                            bar[1], bar[2], bar[3], bar[4]
                        )

                # ════════════ 初回白バー検出後に探索範囲をY軸固定 ════════════
                if bar is not None and not _regions_locked:
                    bar_cy = bar[1] + bar[3] // 2
                    tcx = self._bar_locked_cx or (bar[0] + bar[2] // 2)

                    y_top = max(0, bar_cy - config.REGION_UP)
                    y_bot = min(h_scr, bar_cy + config.REGION_DOWN)
                    _roi = config.DETECT_ROI

                    if _roi:
                        y_top = max(y_top, _roi[1])
                        y_bot = min(y_bot, _roi[1] + _roi[3])

                    rh = y_bot - y_top

                    # 魚探索範囲
                    fish_half = max(config.REGION_X * 2, 80)
                    fsx = max(0, tcx - fish_half)
                    fsw = min(fish_half * 2, w_scr - fsx)
                    if _roi:
                        fsx = max(fsx, _roi[0])
                        fsw = min(fsw, _roi[0] + _roi[2] - fsx)
                    search_region = (fsx, y_top, fsw, rh)

                    # 白バー探索範囲
                    bar_half = config.REGION_X
                    bsx = max(0, tcx - bar_half)
                    bsw = min(bar_half * 2, w_scr - bsx)
                    if _roi:
                        bsx = max(bsx, _roi[0])
                        bsw = min(bsw, _roi[0] + _roi[2] - bsx)
                    bar_search_region = (bsx, y_top, bsw, rh)

                    _regions_locked = True
                    log.info(
                        f"  ★ 探索範囲をロック(白バーY={bar_cy}): "
                        f"Y={y_top}~{y_bot} "
                        f"魚X=±{fish_half} バーX=±{bar_half}"
                        f"{' (ROI切り抜き)' if _roi else ''}"
                    )

                # 魚も同じ軌道Xに合わせる
                if fish is not None:
                    raw_fcx = fish[0] + fish[2] // 2

                    if self._bar_locked_cx is not None:
                        if abs(raw_fcx - self._bar_locked_cx) > _FISH_X_HALF:
                            fish = None
                            self._current_fish_name = ""

                    if fish is not None and self._bar_locked_cx is not None:
                        fish = (
                            self._bar_locked_cx - fish[2] // 2,
                            fish[1], fish[2], fish[3], fish[4]
                        )

                # ════════════ 空間妥当性チェック（Y方向のみ） ════════════
                if fish is not None and bar is not None:
                    fish_cy_check = fish[1] + fish[3] // 2
                    bar_cy_check = bar[1] + bar[3] // 2
                    dist_y = abs(fish_cy_check - bar_cy_check)

                    if dist_y > config.MAX_FISH_BAR_DIST:
                        if frame % 30 == 1:
                            log.warning(
                                f"[⚠ 誤検出] 魚Y={fish_cy_check} バーY={bar_cy_check} "
                                f"距離={dist_y}px > {config.MAX_FISH_BAR_DIST}px"
                            )
                        fish = None
                        bar = None

                # ════════════ 直前10フレーム補完 ════════════
                if fish is None:
                    fish = _get_recent(recent_fish, frame, max_age=5)
                    if fish is not None and frame % 10 == 1:
                        log.info("  ↺ 魚を直前10フレーム履歴で補完しました")

                if bar is None:
                    bar = _get_recent(recent_bar, frame, max_age=5)
                    if bar is not None and frame % 10 == 1:
                        log.info("  ↺ 白バーを直前10フレーム履歴で補完しました")

                if _yolo_progress is None:
                    _yolo_progress = _get_recent(recent_progress, frame, max_age=5)
                    if _yolo_progress is not None and frame % 10 == 1:
                        log.info("  ↺ 進捗バーを直前10フレーム履歴で補完しました")

                if _yolo_hook is None:
                    _yolo_hook = _get_recent(recent_hook, frame, max_age=5)
                    if _yolo_hook is not None and frame % 10 == 1:
                        log.info("  ↺ hookを直前10フレーム履歴で補完しました")

                # 最終観測保持（教師データ記録用）
                if fish is not None:
                    self._pd_last_fish_box = fish
                if bar is not None:
                    self._pd_last_bar_box = bar

                # ════════════ デバッグ描画 ════════════
                if not self._need_rotation:
                    self._show_debug_overlay(
                        screen_raw, fish, bar, search_region,
                        bar_search_region=bar_search_region,
                        progress=_yolo_progress,
                        hook=_yolo_hook,
                        status_text=f"🐟 ミニゲーム F{frame:04d}"
                    )
                else:
                    self._show_debug_overlay(
                        screen_raw,
                        bar_search_region=bar_search_region,
                        progress=_yolo_progress,
                        hook=_yolo_hook,
                        status_text=f"🐟 ミニゲーム F{frame:04d} (回転{self._track_angle:.0f}°補正中)"
                    )

                # ════════════ 進捗取得（終了判定に使うが直接終了には使わない） ════════════
                green = 0.0
                hook_distance = None

                if frame <= _PROGRESS_SKIP_FRAMES:
                    pass

                elif _use_yolo and _yolo_progress is not None and _yolo_hook is not None:
                    # progress + hook の両方がある場合は、それを優先
                    hook_distance = self._calc_hook_distance(_yolo_progress, _yolo_hook)
                    green = self._calc_hook_progress_ratio(_yolo_progress, _yolo_hook)

                    if frame % 15 == 0:
                        log.info(
                            f"[HOOK] dist={hook_distance:.1f}px "
                            f"ratio={green:.1%}"
                        )

                    if not self._progress_debug_saved and green > 0:
                        self._progress_debug_saved = True

                        px, py, pw, ph = _yolo_progress[:4]
                        hx, hy, hw, hh = _yolo_hook[:4]

                        _pad = 20
                        _dx = max(0, min(px, hx) - _pad)
                        _dy = max(0, min(py, hy) - _pad)
                        _right = min(w_scr, max(px + pw, hx + hw) + _pad)
                        _bottom = min(h_scr, max(py + ph, hy + hh) + _pad)
                        _dw = _right - _dx
                        _dh = _bottom - _dy

                        if _dw > 0 and _dh > 0:
                            _dbg = screen[_dy:_dy + _dh, _dx:_dx + _dw].copy()

                            cv2.rectangle(
                                _dbg,
                                (px - _dx, py - _dy),
                                (px - _dx + pw, py - _dy + ph),
                                (0, 255, 0), 1
                            )
                            cv2.rectangle(
                                _dbg,
                                (hx - _dx, hy - _dy),
                                (hx - _dx + hw, hy - _dy + hh),
                                (0, 0, 255), 1
                            )

                            hook_center_y = int(hy + hh * 0.5) - _dy
                            progress_bottom_y = (py + ph) - _dy

                            cv2.line(
                                _dbg,
                                (0, hook_center_y),
                                (_dw - 1, hook_center_y),
                                (0, 0, 255), 1
                            )
                            cv2.line(
                                _dbg,
                                (0, progress_bottom_y),
                                (_dw - 1, progress_bottom_y),
                                (0, 255, 255), 1
                            )

                            _info = f"dist={hook_distance:.1f}px ratio={green:.1%}"
                            cv2.putText(
                                _dbg, _info, (2, 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (0, 255, 255), 1
                            )

                            _ddir = os.path.join(config.BASE_DIR, "debug")
                            os.makedirs(_ddir, exist_ok=True)
                            cv2.imwrite(
                                os.path.join(_ddir, "progress_hook_debug.png"),
                                _dbg
                            )

                elif _use_yolo and _yolo_progress is not None:
                    # hookが無い場合だけ旧式の progress 推定へフォールバック
                    green = self.yolo.detect_progress_fill_ratio(screen, _yolo_progress)

                    if not self._progress_debug_saved and green > 0:
                        self._progress_debug_saved = True

                        px, py, pw, ph = _yolo_progress[:4]
                        _pad = 20
                        _dx = max(0, px - _pad)
                        _dy = max(0, py)
                        _dw = min(pw + _pad * 2, w_scr - _dx)
                        _dh = min(ph, h_scr - _dy)

                        if _dw > 0 and _dh > 0:
                            _dbg = screen[_dy:_dy + _dh, _dx:_dx + _dw].copy()
                            cv2.rectangle(
                                _dbg,
                                (px - _dx, 0),
                                (px - _dx + pw, ph),
                                (0, 255, 0),
                                1
                            )
                            _info = f"green={green:.0%}"
                            cv2.putText(
                                _dbg, _info, (2, 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (0, 255, 255), 1
                            )
                            _ddir = os.path.join(config.BASE_DIR, "debug")
                            os.makedirs(_ddir, exist_ok=True)
                            cv2.imwrite(
                                os.path.join(_ddir, "progress_strip.png"),
                                _dbg
                            )

                else:
                    # テンプレートモード時の緑進捗推定
                    _sr_for_progress = search_region

                    if bar is not None:
                        bcx = bar[0] + bar[2] // 2
                        bcy = bar[1] + bar[3] // 2

                        _pr_half_x = max(config.REGION_X * 2, 80)
                        _pr_x = max(0, bcx - _pr_half_x)
                        _pr_y = max(0, bcy - config.REGION_UP)
                        _pr_w = min(_pr_half_x * 2, w_scr - _pr_x)
                        _pr_h = min(config.REGION_UP + config.REGION_DOWN, h_scr - _pr_y)

                        _sr_for_progress = (_pr_x, _pr_y, _pr_w, _pr_h)
                        _last_progress_sr = _sr_for_progress

                    elif _last_progress_sr is not None:
                        _sr_for_progress = _last_progress_sr

                    green = self._check_progress(screen, fish, _sr_for_progress)

                # 急落だけ平滑化（急上昇は成功の可能性があるので潰さない）
                if green < _prev_green and (_prev_green - green) > 0.30:
                    log.debug(
                        f"  進捗が急落 {_prev_green:.0%}→{green:.0%} のため平滑化します"
                    )
                    green = _prev_green * 0.7 + green * 0.3

                if green > 0:
                    _prev_green = green
                if green > _last_green:
                    _last_green = green

                green_history.append(float(green))
                self._pd_last_progress = _last_green
                # ════════════ ミニゲーム終了判定 ════════════
                obj_count = ((fish is not None) + (bar is not None) + (1 if track_alive else 0))

                # 1) 魚+バーの両方が未検出
                if fish is None and bar is None:
                    no_detect += 1

                    # 一定時間見失ったら押しっぱなし解除
                    if no_detect > 5 and not config.IL_RECORD:
                        self.input.mouse_up()

                    if no_detect == 10:
                        log.warning(f"[⚠ ロスト] {no_detect}フレーム連続で魚+バーを検出できません")
                        self.screen.save_debug(screen, "minigame_lost")

                    if no_detect > config.TRACK_LOST_LIMIT:
                        log.info(f"[📋 終了] {no_detect}フレーム連続で有効UIを検出できなかったため終了します")
                        end_reason = "track_lost_limit"
                        break

                    time.sleep(config.GAME_LOOP_INTERVAL)
                    continue
                else:
                    if no_detect > 5:
                        log.info(f"[✓ 復帰] 有効UIを再検出しました（直前は {no_detect} フレームロスト）")
                    no_detect = 0

                # 2) 魚だけの個別ロスト追跡
                if fish is None:
                    fish_lost += 1
                    if fish_gone_since is None:
                        fish_gone_since = time.time()

                    if fish_lost == 30:
                        log.warning(f"[⚠ 魚ロスト] {fish_lost}フレーム連続で魚を検出できません")

                    if had_good_detection and fish_lost > config.FISH_LOST_LIMIT:
                        log.info(f"[📋 終了] 魚を {fish_lost} フレーム見失ったため、ゲーム終了とみなします")
                        end_reason = "fish_lost_limit"
                        break
                else:
                    fish_lost = 0
                    fish_gone_since = None
                    had_good_detection = True

                # 白バー個別ロストの開始時刻管理
                if bar is None:
                    if bar_gone_since is None:
                        bar_gone_since = time.time()
                else:
                    bar_gone_since = None

                # 3) 単体ロスト時間超過判定
                _timeout = config.SINGLE_OBJ_TIMEOUT
                now_t = time.time()

                if (had_good_detection and fish_gone_since is not None
                        and now_t - fish_gone_since > _timeout):
                    elapsed = now_t - fish_gone_since
                    log.info(
                        f"[📋 失敗] 魚が {elapsed:.1f}s 連続で消失しました "
                        f"(>{_timeout}s)。ミニゲーム終了"
                    )
                    end_reason = "fish_gone_timeout"
                    break

                if (had_good_detection and bar_gone_since is not None
                        and now_t - bar_gone_since > _timeout):
                    elapsed = now_t - bar_gone_since
                    log.info(
                        f"[📋 失敗] 白バーが {elapsed:.1f}s 連続で消失しました "
                        f"(>{_timeout}s)。ミニゲーム終了"
                    )
                    end_reason = "bar_gone_timeout"
                    break

                # 4) 検出対象不足
                # config.OBJ_MIN_COUNT 個未満しか見えていない状態が続いたら終了
                if obj_count < config.OBJ_MIN_COUNT:
                    obj_gone_count += 1

                    if obj_gone_count == 1 or obj_gone_count % 10 == 0:
                        has_f = "魚✓" if fish is not None else "魚✗"
                        has_b = "バー✓" if bar is not None else "バー✗"
                        has_t = "軌道✓" if track_alive else "軌道✗"
                        log.warning(
                            f"[⚠ 対象不足] {has_f} {has_b} {has_t} "
                            f"= {obj_count}個 "
                            f"({obj_gone_count}/{config.OBJ_GONE_LIMIT})"
                        )

                    if obj_gone_count >= config.OBJ_GONE_LIMIT:
                        log.info(
                            f"[📋 終了] {obj_gone_count}フレーム連続で "
                            f"検出対象が {obj_count} 個しかなかったため終了します"
                        )
                        end_reason = "obj_gone_limit"
                        break
                else:
                    if obj_gone_count > 3:
                        log.info(
                            f"[✓ 復帰] 検出対象数が {obj_count} に回復しました "
                            f"(直前は {obj_gone_count} フレーム不足)"
                        )
                    obj_gone_count = 0

                # ════════════ 制御（録画 / モデル / PD） ════════════
                if _skip_fish:
                    # ホワイトリスト外魚は放棄
                    self.input.mouse_up()
                    held = False

                elif config.IL_RECORD:
                    # 録画モード
                    self._il_record_frame(frame, fish, bar)
                    held = False

                elif config.IL_USE_MODEL and self._il_policy is not None:
                    # 行動クローニングモデル制御
                    held = self._il_model_control(fish, bar)

                else:
                    # 通常PD制御
                    held = self._control_mouse(fish, bar, search_region)

                if held:
                    hold_count += 1

                # 一定時間後にユーザー設定の debug mode に戻す
                if frame == 50:
                    self.detector.debug_report = self.debug_mode

                # 30フレームごとにログ
                if frame % 30 == 0:
                    fname = self._current_fish_name.replace("fish_", "") \
                        if self._current_fish_name else ""
                    fi = f"魚[{fname}]Y={fish[1]+fish[3]//2}" if fish else "魚=なし"
                    bi = f"バーY={bar[1]+bar[3]//2}" if bar else "バー=なし"
                    vel = f"v={self._bar_velocity:+.0f}"
                    hook_info = (
                        f" | hook:{hook_distance:.1f}px"
                        if hook_distance is not None else ""
                    )
                    log.info(
                        f"[F{frame:04d}] {fi} | {bi} | {vel} | "
                        f"押下:{hold_count} | 進捗:{green:.0%}{hook_info}"
                    )

                time.sleep(config.GAME_LOOP_INTERVAL)

        finally:
            log.info(f"[END_REASON] {end_reason}")

            # ── progress履歴取得 ──
            last_values = list(green_history)

            if last_values:
                # ★ 最後20%のフレームを切り捨て
                cut = int(len(last_values) * 0.2)
                trimmed = last_values[:-cut] if cut > 0 else last_values

                avg_green = sum(trimmed) / len(trimmed)
                max_green = max(trimmed)

                # デバッグログ
                progress_log = ", ".join(f"{v:.1%}" for v in trimmed)
                log.info(
                    f"[PROGRESS_HISTORY] total={len(last_values)} "
                    f"used={len(trimmed)} cut={cut} | [{progress_log}]"
                )
            else:
                avg_green = 0.0
                max_green = 0.0
                log.info("[PROGRESS_HISTORY] frames=0")

            # ── 成功判定 ──
            if _skip_fish:
                success = False
                log.info(
                    f"[⏭ スキップ] 対象外の魚 (avg={avg_green:.0%}, max={max_green:.0%})"
                )
            elif avg_green > config.SUCCESS_PROGRESS:
                success = True
                log.info(
                    f"[✅ 成功] avg {avg_green:.0%} > {config.SUCCESS_PROGRESS:.0%} "
                    f"(max={max_green:.0%})"
                )
            else:
                success = False
                log.info(
                    f"[❌ 失敗] avg {avg_green:.0%} <= {config.SUCCESS_PROGRESS:.0%} "
                    f"(max={max_green:.0%})"
                )
            # ── RL終端学習 ──
            if self._rl is not None:
                terminal_reward = (
                    getattr(config, "RL_SUCCESS_REWARD", 4.0)
                    if success else
                    getattr(config, "RL_FAIL_REWARD", -4.0)
                )
                self._rl_terminal_reward = terminal_reward
                self._rl_episode_reward += terminal_reward

                dummy_next_state = (
                    self._rl_prev_state
                    if self._rl_prev_state is not None
                    else (0, 0, 0, 0, 0, 0)
                )
                self._rl.update(terminal_reward, dummy_next_state, done=True)
                self._rl.save()

                log.info(
                    f"[RL_END] step_sum={self._rl_step_reward_sum:+.4f} "
                    f"terminal={self._rl_terminal_reward:+.4f} "
                    f"total={self._rl_episode_reward:+.4f} "
                    f"success={success} end_reason={end_reason}"
                )
            # PD教師データのエピソード終端記録
            if config.PD_RECORD and (not config.IL_RECORD) and (not config.IL_USE_MODEL):
                try:
                    self._pd_record_episode_end(
                        fish_box=self._pd_last_fish_box,
                        bar_box=self._pd_last_bar_box,
                        fish_name=self._current_fish_name or "",
                        progress=_last_green,
                        success=success,
                        end_reason=end_reason,
                    )
                except Exception as e:
                    log.warning(f"[PD_RECORD] episode_end 書き込み失敗: {e}")

            # 録画モードなら手動収竿
            if config.IL_RECORD:
                self._il_stop_recording()
                log.info("[🎣 収竿] 録画モードのため手動で収竿してください")
            else:
                # 念のため入力解放
                self.input.safe_release()
                time.sleep(config.POST_RELEASE_DELAY)

                if success:
                    time.sleep(config.SUCCESS_CLICK_DELAY)
                    self.input.click()
                    log.info(
                        f"[🎣 収竿] 釣り成功。"
                        f"{config.SUCCESS_CLICK_DELAY:.2f}s 後にクリックして収竿しました"
                    )
                else:
                    log.info("[🎣 失敗] 竿は自動で戻っているため、収竿クリックは行いません")

                    # 失敗時だけ画像収集したい場合
                    if config.YOLO_COLLECT_ON_FAIL:
                        try:
                            _cdir = os.path.join(
                                config.BASE_DIR, "yolo", "dataset",
                                "images", "unlabeled"
                            )
                            os.makedirs(_cdir, exist_ok=True)
                            _ts = time.strftime("%Y%m%d_%H%M%S")
                            _ms = int((time.time() % 1) * 1000)
                            _fail_screen = self._grab()

                            cv2.imwrite(
                                os.path.join(_cdir, f"fail_{_ts}_{_ms:03d}.png"),
                                _fail_screen
                            )
                            log.info(f"[YOLO] 失敗時画像を保存しました: fail_{_ts}_{_ms:03d}.png")
                        except Exception as e:
                            log.warning(f"[YOLO] 失敗時画像保存中に例外: {e}")

        return success

    # ══════════════════════════════════════════════════════
    #  可視化デバッグ
    # ══════════════════════════════════════════════════════

    def _calc_hook_distance(self, progress_box, hook_box):
        """
        progress下端 → hook中心までの縦距離(px)を返す

        Returns
        -------
        float | None
        """
        if progress_box is None or hook_box is None:
            return None

        px, py, pw, ph = progress_box[:4]
        hx, hy, hw, hh = hook_box[:4]

        progress_bottom = py + ph
        hook_center_y = hy + hh * 0.5

        dist = progress_bottom - hook_center_y
        if dist < 0:
            dist = 0.0

        return float(dist)

    def _calc_hook_progress_ratio(self, progress_box, hook_box):
        """
        progress下端 → hook中心までの距離を
        progress高さで正規化して 0.0 ~ 1.0 で返す
        """
        if progress_box is None or hook_box is None:
            return 0.0

        _, _, _, ph = progress_box[:4]
        if ph <= 1:
            return 0.0

        dist = self._calc_hook_distance(progress_box, hook_box)
        if dist is None:
            return 0.0

        return float(max(0.0, min(1.0, dist / ph)))

    def _show_debug_overlay(self, screen, fish=None, bar=None,
                            search_region=None, bar_search_region=None,
                            progress=None, hook=None, status_text=""):
        """
        統一デバッグウィンドウ。
        すべての段階で使える。

        ★ 先に縮小画像を作ってから重ね描画することで、
          CPU / メモリ負荷を大きく下げている。
        """
        if not config.SHOW_DEBUG:
            return

        now = time.time()
        if now - self._last_overlay_time < config.DEBUG_OVERLAY_INTERVAL:
            return
        self._last_overlay_time = now

        # ── ROI切り抜き: 選択範囲だけ表示 ──
        _roi = config.DETECT_ROI
        ox, oy = 0, 0
        if _roi:
            rx, ry, rw, rh = _roi
            sh, sw = screen.shape[:2]
            rx = max(0, min(rx, sw - 1))
            ry = max(0, min(ry, sh - 1))
            rw = min(rw, sw - rx)
            rh = min(rh, sh - ry)
            if rw > 20 and rh > 20:
                screen = screen[ry:ry + rh, rx:rx + rw].copy()
                ox, oy = rx, ry

        h, w = screen.shape[:2]
        max_w = config.DEBUG_OVERLAY_MAX_W
        max_h = config.DEBUG_OVERLAY_MAX_H
        s = min(max_w / w, max_h / h, 1.0)

        if s < 1.0:
            debug = cv2.resize(
                screen, (int(w * s), int(h * s)),
                interpolation=cv2.INTER_NEAREST
            )
        else:
            debug = screen.copy()
            s = 1.0

        # 座標縮尺補助関数（ROIオフセットを引いてから拡大縮小）
        def sx(v):
            return int((v - ox) * s)

        def sy(v):
            return int((v - oy) * s)

        # ── 上部ステータス文字 ──
        y_txt = 22
        fs = 0.55
        dw = debug.shape[1]

        # FPS表示
        fps_text = f"{self._fps:.1f} FPS"
        fps_color = (
            (0, 255, 0) if self._fps >= 10
            else (0, 255, 255) if self._fps >= 5
            else (0, 0, 255)
        )
        cv2.putText(
            debug, fps_text, (dw - 120, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2
        )

        if status_text:
            cv2.putText(
                debug, status_text, (8, y_txt),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 255), 1
            )
            y_txt += 22

        if self._need_rotation:
            cv2.putText(
                debug, f"Rotation: {-self._track_angle:.1f} deg",
                (8, y_txt), cv2.FONT_HERSHEY_SIMPLEX, fs,
                (0, 200, 255), 1
            )
            y_txt += 20

        # 制御状態 + 速度表示
        if fish is not None and bar is not None:
            fish_cy = fish[1] + fish[3] // 2
            bar_cy  = bar[1] + bar[3] // 2
            diff = bar_cy - fish_cy

            if diff > config.DEAD_ZONE:
                label = f"v BAR below (d={diff}px)"
                lcolor = (0, 100, 255)
            elif diff < -config.DEAD_ZONE:
                label = f"^ BAR above (d={diff}px)"
                lcolor = (255, 200, 0)
            else:
                label = f"= dead zone (d={diff}px)"
                lcolor = (0, 255, 0)

            cv2.putText(
                debug, label, (8, y_txt),
                cv2.FONT_HERSHEY_SIMPLEX, fs, lcolor, 1
            )
            y_txt += 20

        elif fish is None and bar is None and self.state == "ミニゲーム中":
            cv2.putText(
                debug, "X no fish+bar", (8, y_txt),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 255), 1
            )
            y_txt += 20

        if abs(self._bar_velocity) > 0.5:
            cv2.putText(
                debug, f"v={self._bar_velocity:+.0f} px/s",
                (8, y_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1
            )
            y_txt += 18

        # ── 探索範囲描画（灰色=魚、薄シアン=白バー） ──
        if search_region:
            rx, ry, rw, rh = [int(v) for v in search_region]
            cv2.rectangle(
                debug, (sx(rx), sy(ry)),
                (sx(rx + rw), sy(ry + rh)),
                (128, 128, 128), 1
            )

        if bar_search_region:
            bx, by, bw, bh = [int(v) for v in bar_search_region]
            cv2.rectangle(
                debug, (sx(bx), sy(by)),
                (sx(bx + bw), sy(by + bh)),
                (128, 200, 200), 1
            )

        # ── 魚描画 + 名前表示 ──
        if fish is not None:
            fx, fy, fw, fh = fish[:4]
            fish_cy = fy + fh // 2
            fname, fcolor = self.FISH_DISPLAY.get(
                self._current_fish_name, ("?", (0, 255, 0))
            )

            cv2.rectangle(
                debug, (sx(fx), sy(fy)),
                (sx(fx + fw), sy(fy + fh)),
                fcolor, 2
            )
            cv2.putText(
                debug, f"{fname} Y={fish_cy}",
                (sx(fx + fw) + 4, sy(fish_cy)),
                cv2.FONT_HERSHEY_SIMPLEX, fs, fcolor, 1
            )
            cv2.line(
                debug, (sx(fx), sy(fish_cy)),
                (sx(fx + fw), sy(fish_cy)),
                fcolor, 1
            )

        # ── 白バー描画（青） ──
        if bar is not None:
            bx, by, bw, bh = bar[:4]
            bar_cy = by + bh // 2
            cv2.rectangle(
                debug, (sx(bx), sy(by)),
                (sx(bx + bw), sy(by + bh)),
                (255, 100, 0), 2
            )
            cv2.putText(
                debug, f"Bar Y={bar_cy}",
                (max(0, sx(bx) - 90), sy(bar_cy)),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 100, 0), 1
            )
            cv2.line(
                debug, (sx(bx), sy(bar_cy)),
                (sx(bx + bw), sy(bar_cy)),
                (255, 100, 0), 1
            )

        # ── 進捗バー描画 ──
        if progress is not None:
            px, py, pw, ph = progress[:4]
            cv2.rectangle(
                debug, (sx(px), sy(py)),
                (sx(px + pw), sy(py + ph)),
                (0, 220, 180), 2
            )
            cv2.putText(
                debug, "Progress",
                (sx(px), sy(py) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 180), 1
            )

        # ── hook描画 ──
        if hook is not None:
            hx, hy, hw, hh = hook[:4]
            hook_cy = hy + hh // 2
            cv2.rectangle(
                debug, (sx(hx), sy(hy)),
                (sx(hx + hw), sy(hy + hh)),
                (0, 0, 255), 2
            )
            cv2.putText(
                debug, "Hook",
                (sx(hx), sy(hy) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1
            )
            cv2.line(
                debug, (sx(hx), sy(hook_cy)),
                (sx(hx + hw), sy(hook_cy)),
                (0, 0, 255), 1
            )

        # ── 魚と白バーの距離線 ──
        if fish is not None and bar is not None:
            fish_cy = fish[1] + fish[3] // 2
            bar_cy  = bar[1] + bar[3] // 2
            cx = (fish[0] + bar[0]) // 2
            diff = bar_cy - fish_cy
            color = (0, 0, 255) if abs(diff) > 50 else (0, 255, 255)

            cv2.arrowedLine(
                debug, (sx(cx), sy(bar_cy)),
                (sx(cx), sy(fish_cy)),
                color, 1, tipLength=0.15
            )
            cv2.putText(
                debug, f"d={diff:+d}",
                (sx(cx) + 6, sy((fish_cy + bar_cy) // 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1
            )

        with self._debug_lock:
            self._debug_frame = debug

        if self._debug_thread is None or not self._debug_thread.is_alive():
            self._debug_thread = threading.Thread(
                target=self._debug_display_loop, daemon=True
            )
            self._debug_thread.start()

    def _debug_display_loop(self):
        """
        別スレッド側:
        debugフレームをループ表示する。
        cv2.waitKey の待ちが釣り本体を止めないように分離している。
        """
        while self.running or self._debug_frame is not None:
            frame = None
            with self._debug_lock:
                if self._debug_frame is not None:
                    frame = self._debug_frame
                    self._debug_frame = None

            if frame is not None:
                try:
                    cv2.imshow("Debug Overlay", frame)
                except Exception:
                    break

            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break

        try:
            cv2.destroyWindow("Debug Overlay")
        except Exception:
            pass

    # ══════════════════════════════════════════════════════
    #  ミニゲーム補助
    # ══════════════════════════════════════════════════════

    def _init_search_region(self, screen):
        """
        初期探索範囲を決定し、(fish_region, track_center_x, bar_region) を返す。

        ルール:
        - DETECT_ROI が設定されている場合:
            ROI内だけで軌道 / 白バーを探索
            ROI自体を探索範囲として使う
        - ROI が無い場合:
            白バー + 軌道の交差確認で位置を絞り込む
        """
        h, w = screen.shape[:2]
        roi = config.DETECT_ROI

        # ROI有効性チェック
        if roi:
            rx, ry, rw, rh = roi
            if rx + rw > w or ry + rh > h or rw < 20 or rh < 20:
                log.warning(
                    f"  ► ROI ({rx},{ry},{rw},{rh}) が画面 "
                    f"({w}x{h}) をはみ出すか小さすぎるため無視します"
                )
                roi = None

        # ROI（または全画面）内で白バーと軌道を探す
        bar = self.detector.find_multiscale(
            screen, "bar", config.THRESH_BAR,
            scales=config.BAR_SCALES,
            search_region=roi,
        )
        track = self.detector.find_multiscale(
            screen, "track", config.THRESH_TRACK,
            search_region=roi,
        )

        bar_cx = (bar[0] + bar[2] // 2) if bar else None
        track_cx = (track[0] + track[2] // 2) if track else None

        chosen_cx = None

        if bar_cx is not None and track_cx is not None:
            if abs(bar_cx - track_cx) < 150:
                chosen_cx = bar_cx
                log.info(
                    f"  ► 軌道+白バー一致: 軌道X={track_cx}(conf={track[4]:.2f}) "
                    f"白バーX={bar_cx}(conf={bar[4]:.2f}) → 白バーXを採用"
                )
            else:
                chosen_cx = bar_cx
                log.warning(
                    f"  ► 軌道X={track_cx}(conf={track[4]:.2f}) "
                    f"白バーX={bar_cx}(conf={bar[4]:.2f}) が不一致、"
                    f"白バー基準にします"
                )

        elif bar_cx is not None:
            chosen_cx = bar_cx
            log.info(f"  ► 白バーのみ検出 @ X={bar_cx} conf={bar[4]:.2f}")

        elif track_cx is not None:
            chosen_cx = track_cx
            log.info(f"  ► 軌道のみ検出 @ X={track_cx} conf={track[4]:.2f}")

        # ROIがある場合はそのまま使う
        if roi:
            roi_t = tuple(roi)
            if chosen_cx is None:
                chosen_cx = roi[0] + roi[2] // 2
                log.info(f"  ► ROI内で軌道/白バー未検出。ROI中心 X={chosen_cx} を使用")
            log.info(
                f"  ★ 選択範囲使用: X={roi[0]} Y={roi[1]} "
                f"{roi[2]}x{roi[3]}"
            )
            return roi_t, chosen_cx, roi_t

        # ROIが無い場合は検出結果から範囲構築
        if chosen_cx is not None:
            y_start = h // 3

            bar_half = max(config.REGION_X, 60)
            bsx = max(0, chosen_cx - bar_half)
            bsw = min(bar_half * 2, w - bsx)
            bar_region = (bsx, y_start, bsw, h - y_start)

            fish_half = max(config.REGION_X * 2, 120)
            fsx = max(0, chosen_cx - fish_half)
            fsw = min(fish_half * 2, w - fsx)
            fish_region = (fsx, y_start, fsw, h - y_start)

            return fish_region, chosen_cx, bar_region

        # 最後のフォールバック
        sw = int(w * 0.6)
        y_start = h // 2
        log.info("  ► 軌道も白バーも未検出のため、左側下半分を使用")
        fallback = (0, y_start, sw, h - y_start)
        return fallback, None, fallback

    _progress_debug_saved = False

    def _check_progress(self, screen, fish, sr):
        """
        進捗バーの緑色部分を検出する。

        方法:
        白バー中心Xの左側にある 5px 幅の細い縦帯だけを見て、
        緑色割合を測る。
        背景ノイズの影響を減らすための工夫。
        """
        if sr is None:
            return 0.0

        bar_cx = self._bar_locked_cx
        if bar_cx is None:
            if fish is not None:
                bar_cx = fish[0]
            else:
                bar_cx = sr[0] + sr[2] // 3

        strip_w = 5
        sx = max(0, bar_cx - strip_w - 8)
        sy = sr[1]
        sw = strip_w
        sh = sr[3]

        if sx + sw > screen.shape[1]:
            sw = screen.shape[1] - sx
        if sy + sh > screen.shape[0]:
            sh = screen.shape[0] - sy
        if sw <= 0 or sh <= 0:
            return 0.0

        ratio = self.detector.detect_green_ratio(
            screen, (sx, sy, sw, sh)
        )

        if not self._progress_debug_saved and ratio > 0:
            self._progress_debug_saved = True
            import os
            pad = 30
            dx = max(0, sx - pad)
            dw = min(sw + pad * 2, screen.shape[1] - dx)
            dbg = screen[sy:sy + sh, dx:dx + dw].copy()

            cv2.rectangle(
                dbg, (sx - dx, 0), (sx - dx + sw, sh),
                (0, 255, 0), 1
            )

            info = f"green={ratio:.0%} w={strip_w}"
            cv2.putText(
                dbg, info, (2, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1
            )

            debug_dir = os.path.join(config.BASE_DIR, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(
                os.path.join(debug_dir, "progress_strip.png"), dbg
            )

        return ratio

    # ══════════════════════════════════════════════════════
    #  行動クローニング: 録画 / 推論
    # ══════════════════════════════════════════════════════

    def _load_il_policy(self):
        """学習済み行動クローニングモデルを読み込む（正規化パラメータ込み）"""
        try:
            import torch
            from imitation.model import FishPolicy

            checkpoint = torch.load(
                config.IL_MODEL_PATH,
                map_location="cpu",
                weights_only=True
            )

            # 旧形式（state_dictのみ）と新形式（正規化込み）両対応
            if "model_state" in checkpoint:
                state = checkpoint["model_state"]
                self._il_norm_mean = checkpoint["norm_mean"].numpy()
                self._il_norm_std = checkpoint["norm_std"].numpy()
                hist_len = checkpoint.get("history_len", config.IL_HISTORY_LEN)
            else:
                state = checkpoint
                self._il_norm_mean = None
                self._il_norm_std = None
                hist_len = config.IL_HISTORY_LEN

            model = FishPolicy(history_len=hist_len)
            model.load_state_dict(state)
            model.eval()

            if torch.cuda.is_available():
                model = model.cuda()
                self._il_device = "cuda"

            self._il_policy = model
            norm_info = "正規化あり" if self._il_norm_mean is not None else "正規化なし"
            log.info(f"[IL] モデル読み込み完了 ({self._il_device}, {norm_info})")

        except Exception as e:
            log.warning(f"[IL] モデル読み込み失敗: {e}")
            self._il_policy = None

    def _il_start_recording(self):
        """1回分のミニゲームデータ録画を開始する"""
        os.makedirs(config.IL_DATA_DIR, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(config.IL_DATA_DIR, f"session_{ts}.csv")

        self._il_file = open(path, "w", newline="", encoding="utf-8")
        self._il_writer = csv.writer(self._il_file)
        self._il_writer.writerow([
            "frame", "timestamp",
            "fish_cy", "bar_cy", "bar_h",
            "error", "velocity", "fish_delta", "dist_ratio",
            "mouse_pressed",
            "fish_in_bar", "press_streak",
            "predicted", "bar_accel",
        ])

        self._il_prev_fish_cy = None
        self._il_mouse_prev = 0
        self._il_history.clear()

        log.info(f"[IL] 録画開始 → {path}")

    def _il_stop_recording(self):
        """録画終了"""
        if self._il_file:
            self._il_file.close()
            self._il_file = None
            self._il_writer = None
            log.info("[IL] 録画終了")

    @staticmethod
    def _is_mouse_pressed() -> bool:
        """ユーザーが左クリックを押し続けているか判定する"""
        return ctypes.windll.user32.GetAsyncKeyState(0x01) & 0x8000 != 0

    def _il_build_features(self, fish, bar):
        """検出結果から1フレーム分の特徴量 [10次元] を構築する"""
        fish_cy = fish[1] + fish[3] // 2
        bar_cy = bar[1] + bar[3] // 2
        bar_h = bar[3]
        bar_top = bar[1]

        error = bar_cy - fish_cy
        velocity = self._bar_velocity

        fish_delta = 0.0
        if self._il_prev_fish_cy is not None:
            fish_delta = fish_cy - self._il_prev_fish_cy
        self._il_prev_fish_cy = fish_cy

        dist_ratio = error / max(bar_h, 1)
        fish_in_bar = (fish_cy - bar_top) / max(bar_h, 1)

        if self._il_mouse_prev == 1:
            self._il_press_streak = max(1, getattr(self, '_il_press_streak', 0) + 1)
        else:
            self._il_press_streak = min(-1, getattr(self, '_il_press_streak', 0) - 1)

        press_streak = self._il_press_streak / 10.0

        # 慣性予測: 150ms 後の白バー相対位置
        predicted = error + velocity * 0.15

        # 加速度 = 速度変化量
        bar_accel = 0.0
        if hasattr(self, '_il_prev_velocity'):
            bar_accel = velocity - self._il_prev_velocity
        self._il_prev_velocity = velocity

        return [
            error, velocity, bar_h, fish_delta, dist_ratio,
            self._il_mouse_prev, fish_in_bar, press_streak,
            predicted, bar_accel
        ]

    def _il_record_frame(self, frame_idx, fish, bar):
        """1フレーム録画: ユーザーのマウス状態を読み取ってCSVへ保存する"""
        if fish is None or bar is None or self._il_writer is None:
            return

        mouse = 1 if self._is_mouse_pressed() else 0
        feats = self._il_build_features(fish, bar)

        fish_cy = fish[1] + fish[3] // 2
        bar_cy = bar[1] + bar[3] // 2
        bar_h = bar[3]
        error = feats[0]
        velocity = feats[1]
        fish_delta = feats[3]
        dist_ratio = feats[4]
        fish_in_bar = feats[6]
        press_streak = feats[7]
        predicted = feats[8]
        bar_accel = feats[9]

        self._il_writer.writerow([
            frame_idx, f"{time.time():.4f}",
            fish_cy, bar_cy, bar_h,
            f"{error:.1f}", f"{velocity:.1f}", f"{fish_delta:.1f}",
            f"{dist_ratio:.3f}",
            mouse,
            f"{fish_in_bar:.3f}", f"{press_streak:.2f}",
            f"{predicted:.1f}", f"{bar_accel:.1f}",
        ])

        self._il_mouse_prev = mouse

    def _il_model_control(self, fish, bar) -> bool:
        """
        学習済みモデルで押す / 離すを決める。

        出力は「今この瞬間、マウスは押下状態か解放状態か」。
        つまりパルス制御ではなく状態制御。
        """
        import torch
        import numpy as np

        if self._il_policy is None:
            return False

        if fish is not None and bar is not None:
            feats = self._il_build_features(fish, bar)
            self._il_history.append(feats)

        elif fish is None and bar is None:
            self.input.mouse_up()
            self._il_mouse_prev = 0
            return False

        # 履歴不足時は仮で押す
        if len(self._il_history) < config.IL_HISTORY_LEN:
            self.input.mouse_down()
            self._il_mouse_prev = 1
            return True

        flat = []
        for f in self._il_history:
            flat.extend(f)

        flat_np = np.array(flat, dtype=np.float32)

        if self._il_norm_mean is not None:
            flat_np = (flat_np - self._il_norm_mean) / self._il_norm_std

        x = torch.from_numpy(flat_np).unsqueeze(0).to(self._il_device)
        prob = self._il_policy.predict(x)

        fish_cy = fish[1] + fish[3] // 2 if fish else -1
        bar_cy = bar[1] + bar[3] // 2 if bar else -1

        thresh = config.IL_PRESS_THRESH

        if prob > thresh:
            self.input.mouse_down()
            self._il_mouse_prev = 1

            if fish is not None and bar is not None and self._il_log_counter % 10 == 0:
                log.info(
                    f"  [IL] 魚Y={fish_cy} バーY={bar_cy} "
                    f"p={prob:.2f}>{thresh:.2f} → 押下"
                )

            self._il_log_counter += 1
            return True

        else:
            self.input.mouse_up()
            self._il_mouse_prev = 0

            if fish is not None and bar is not None and self._il_log_counter % 10 == 0:
                log.info(
                    f"  [IL] 魚Y={fish_cy} バーY={bar_cy} "
                    f"p={prob:.2f}<={thresh:.2f} → 解放"
                )

            self._il_log_counter += 1
            return False

    @staticmethod
    def _clip(v, lo, hi):
        return max(lo, min(hi, v))

    @staticmethod
    def _bin(v, lo, hi, bins):
        v = max(lo, min(hi, v))
        if hi - lo < 1e-9:
            return 0
        r = (v - lo) / (hi - lo)
        idx = int(r * bins)
        if idx >= bins:
            idx = bins - 1
        if idx < 0:
            idx = 0
        return idx

    def _build_rl_state(self, *, error_px, bar_velocity, fish_delta, fish_in_bar, base_hold, progress):
        """
        Q-table用に状態を粗く離散化する
        """
        return (
            self._bin(error_px,      -180, 180, getattr(config, "RL_ERR_BIN", 12)),
            self._bin(bar_velocity,  -900, 900, getattr(config, "RL_VEL_BIN", 8)),
            self._bin(fish_delta,    -80,   80, getattr(config, "RL_FISHDELTA_BIN", 6)),
            self._bin(fish_in_bar,   -0.5, 1.5, getattr(config, "RL_FIB_BIN", 12)),
            self._bin(base_hold,      0.0, 0.12, getattr(config, "RL_HOLD_BIN", 8)),
            self._bin(progress,       0.0, 1.0, getattr(config, "RL_PROG_BIN", 8)),
        )

    def _calc_rl_step_reward(
        self,
        *,
        abs_error,
        fish_in_bar,
        progress,
        prev_abs_error,
        prev_delta_hold=0.0,
    ):
        reward = 0.0

        prev_progress = self._rl_prev_progress

        in_bar = (0.0 <= fish_in_bar <= 1.0)
        prev_in_bar = getattr(self, "_rl_prev_in_bar", True)

        # 1) 生存ボーナス
        reward += getattr(config, "RL_REWARD_ALIVE", 0.005)

        # 2) バー内 / バー外の評価
        if in_bar:
            center_err = abs(fish_in_bar - 0.5) / 0.5   # 0~1
            reward += getattr(config, "RL_REWARD_CENTER_SHAPE", 0.08) * (1.0 - center_err)
            reward += getattr(config, "RL_REWARD_IN_BAR", 0.04)
        else:
            # バー外にいるほど強く罰する
            if fish_in_bar < 0.0:
                outside_dist = -fish_in_bar
            else:
                outside_dist = fish_in_bar - 1.0

            outside_dist = min(outside_dist, 1.5)
            reward -= getattr(config, "RL_REWARD_OUTSIDE_SCALE", 0.18) * outside_dist
            reward -= getattr(config, "RL_REWARD_OUTSIDE_FLAT", 0.04)

        # 3) 誤差改善ボーナス
        if prev_abs_error is not None:
            improve = prev_abs_error - abs_error
            improve = max(-25.0, min(25.0, improve))
            reward += improve * getattr(config, "RL_REWARD_IMPROVE_GAIN", 0.015)

        # 4) 再捕捉ボーナス / 脱落ペナルティ
        if (not prev_in_bar) and in_bar:
            reward += getattr(config, "RL_REWARD_RECAPTURE", 0.35)

        if prev_in_bar and (not in_bar):
            reward -= getattr(config, "RL_REWARD_LOSE_BAR", 0.18)

        # 5) progress差分
        dprog = progress - prev_progress
        dprog = max(-0.08, min(0.08, dprog))
        reward += dprog * getattr(config, "RL_REWARD_PROGRESS_GAIN", 0.60)

        # 6) 行動ペナルティ
        # 外れている時は大胆に補正してよいので軽くする
        action_penalty = getattr(config, "RL_REWARD_ACTION_PENALTY", 0.35)
        if not in_bar:
            action_penalty *= 0.35

        reward -= abs(prev_delta_hold) * action_penalty

        # 次フレーム用
        self._rl_prev_in_bar = in_bar

        return float(reward)
    
    def _control_mouse(self, fish, bar, sr) -> bool:
        """
        高品質PD + residual RL
        RLは base_hold に delta_hold を足すだけ。
        """
        now = time.time()

        # ── 白バー速度推定 ──
        if bar is not None:
            bar_cy_raw = bar[1] + bar[3] // 2

            if (self._bar_prev_cy is not None
                    and self._bar_prev_time is not None):
                dt = now - self._bar_prev_time
                if dt > 0.003:
                    raw_vel = (bar_cy_raw - self._bar_prev_cy) / dt
                    alpha = min(config.VELOCITY_SMOOTH, 0.95)
                    self._bar_velocity = (
                        alpha * self._bar_velocity + (1 - alpha) * raw_vel
                    )

            self._bar_prev_cy = bar_cy_raw
            self._bar_prev_time = now

        vel = self._bar_velocity

        TARGET_FIB = 0.5
        KP         = getattr(config, 'HOLD_GAIN', 0.040)
        KD         = getattr(config, 'SPEED_DAMPING', 0.00025)
        BASE_HOLD  = getattr(config, 'HOLD_MIN_S', 0.025)
        MAX_HOLD   = getattr(config, 'HOLD_MAX_S', 0.100)
        MIN_HOLD   = 0.004

        action_press = 0
        control_value = 0.0
        in_deadzone = False

        # ── 魚もバーもある通常ケース ──
        if fish is not None and bar is not None:
            raw_fish_cy = fish[1] + fish[3] // 2
            bar_cy      = bar[1] + bar[3] // 2

            # 魚位置EMA
            if self._fish_smooth_cy is None:
                self._fish_smooth_cy = float(raw_fish_cy)
            else:
                self._fish_smooth_cy = 0.4 * raw_fish_cy + 0.6 * self._fish_smooth_cy

            fish_cy = int(self._fish_smooth_cy)
            bar_h   = max(bar[3], 1)
            bar_top = bar[1]
            fish_in_bar = (fish_cy - bar_top) / bar_h

            error = TARGET_FIB - fish_in_bar
            error_clamp = max(-2.0, min(2.0, error))

            # まずPDだけで base_hold を作る
            if fish_in_bar < 0.0 or fish_in_bar > 1.0:
                kp_use = KP * 1.35
            else:
                kp_use = KP

            base_hold = BASE_HOLD + error_clamp * kp_use + vel * KD
            base_hold = max(MIN_HOLD, min(base_hold, MAX_HOLD))

            fish_delta = 0.0
            if self._last_fish_cy is not None:
                fish_delta = fish_cy - self._last_fish_cy

            abs_error_px = abs(bar_cy - fish_cy)

            final_hold = base_hold
            delta_hold = 0.0

            # ── RL補正 ──
            if self._rl is not None:
                state = self._build_rl_state(
                    error_px=(bar_cy - fish_cy),
                    bar_velocity=vel,
                    fish_delta=fish_delta,
                    fish_in_bar=fish_in_bar,
                    base_hold=base_hold,
                    progress=self._pd_last_progress,
                )

                # 1ステップ前の行動を今の観測で学習
                step_reward = self._calc_rl_step_reward(
                    abs_error=abs_error_px,
                    fish_in_bar=fish_in_bar,
                    progress=self._pd_last_progress,
                    prev_abs_error=self._rl_prev_abs_error,
                    prev_delta_hold=self._rl_prev_delta_hold,
                )
                self._rl_episode_reward += step_reward
                self._rl_step_reward_sum += step_reward
                self._rl.update(step_reward, state, done=False)

                # 今回の補正量を選ぶ
                delta_hold, _ = self._rl.act(state)

                # 補正幅を少し制限（PD補正として暴れにくくする）
                max_delta = getattr(config, "RL_MAX_DELTA_HOLD", 0.020)
                delta_hold = max(-max_delta, min(max_delta, delta_hold))

                final_hold = base_hold + delta_hold
                final_hold = max(MIN_HOLD, min(final_hold, MAX_HOLD))

                self._rl_prev_state = state
                self._rl_prev_progress = self._pd_last_progress
                self._rl_prev_abs_error = abs_error_px
                self._rl_prev_delta_hold = delta_hold

                # ステップ報酬ログ（出しすぎ防止で30フレームごと）
                if getattr(config, "RL_LOG_STEP_REWARD", True):
                    if not hasattr(self, "_rl_log_step_counter"):
                        self._rl_log_step_counter = 0
                    self._rl_log_step_counter += 1

                    if self._rl_log_step_counter % getattr(config, "RL_LOG_STEP_INTERVAL", 30) == 0:
                        log.info(
                            f"[RL_STEP] r={step_reward:+.4f} "
                            f"sum={self._rl_step_reward_sum:+.4f} "
                            f"err_px={bar_cy - fish_cy:+.1f} "
                            f"fib={fish_in_bar:.3f} "
                            f"prog={self._pd_last_progress:.1%} "
                            f"prev_d={self._rl_prev_delta_hold*1000:+.1f}ms"
                        )

            action_press = 1 if final_hold >= MIN_HOLD + 0.001 else 0
            control_value = float(final_hold)
            in_deadzone = abs(error) < 0.05

            self._pd_last_fish_box = fish
            self._pd_last_bar_box = bar

            if config.PD_RECORD and (not config.IL_RECORD) and (not config.IL_USE_MODEL):
                self._pd_record_frame(
                    fish_box=fish,
                    bar_box=bar,
                    fish_name=self._current_fish_name or "",
                    action_press=action_press,
                    control_value=control_value,
                    in_deadzone=in_deadzone,
                    progress=self._pd_last_progress,
                    end_reason="",
                    episode_done=0,
                    episode_success=0,
                )

            self._last_hold = final_hold
            self._last_fish_cy = fish_cy

            fname = (self._current_fish_name.replace("fish_", "")
                     if self._current_fish_name else "?")

            if action_press:
                self.input.mouse_down()
                time.sleep(final_hold)
                self.input.mouse_up()

                if self._rl is not None:
                    log.info(
                        f"  ● [{fname}] fib={fish_in_bar:.2f} v={vel:+.0f} "
                        f"PD={base_hold*1000:.0f}ms RL={delta_hold*1000:+.0f}ms "
                        f"→ {final_hold*1000:.0f}ms"
                    )
                else:
                    log.info(
                        f"  ● [{fname}] fib={fish_in_bar:.2f} v={vel:+.0f} "
                        f"→ 押下 {final_hold*1000:.0f}ms"
                    )
                return True
            else:
                self.input.mouse_up()
                return False

        # ── フォールバック時は従来通り ──
        fallback = self._last_hold
        if fallback is None:
            fallback = BASE_HOLD

        fallback = 0.6 * fallback + 0.4 * BASE_HOLD
        self._last_hold = fallback

        if fish is not None:
            fish_cy = fish[1] + fish[3] // 2
            self._last_fish_cy = fish_cy
            self._pd_last_fish_box = fish

            if sr is not None:
                mid_y = sr[1] + sr[3] // 2
            elif config.DETECT_ROI:
                mid_y = config.DETECT_ROI[1] + config.DETECT_ROI[3] // 2
            else:
                mid_y = fish_cy

            if fish_cy < mid_y:
                h = min(fallback * 1.5, MAX_HOLD)
                self.input.mouse_down()
                time.sleep(h)
                self.input.mouse_up()
                return True
            else:
                self.input.mouse_up()
                return False

        elif bar is not None:
            self._pd_last_bar_box = bar
            self.input.mouse_down()
            time.sleep(fallback)
            self.input.mouse_up()
            return True

        return False

    # ══════════════════════════════════════════════════════
    #  メインループ（バックグラウンドスレッドで実行）
    # ══════════════════════════════════════════════════════

    def run(self):
        """メイン釣りループ — GUI側からバックグラウンドスレッドで起動される"""
        log.info("釣りスレッドを開始しました")

        while self.running:
            try:
                if config.IL_RECORD:
                    # 録画モード: ユーザーが手動操作し、プログラムはミニゲームUI出現待ち
                    self.state = "録画: ミニゲーム待機"
                    log.info("[IL] 手動で 投竿→待機→合わせ を行ってください。ミニゲーム出現待ちです...")
                    if not self._wait_for_minigame_ui():
                        break

                else:
                    self._cast_rod()
                    if not self.running:
                        break

                    if not self._wait_for_bite():
                        if self.running:
                            time.sleep(1.0)
                        continue
                    if not self.running:
                        break

                    self._hook_fish()
                    if not self.running:
                        break

                    # 本当にミニゲームが出たか確認
                    if not self._verify_minigame():
                        log.info("[❌ 未検出] ミニゲーム未検出 → 失敗として扱います")

                        result = False
                        self.fish_count += 1
                        self.fail_count += 1
                        self._consecutive_fail += 1

                        # 未検出ベースで頭向き補正更新
                        self._update_section_head_adjust(True)

                        log.info(
                            f"[🎣 結果] 第 {self.fish_count} 回釣り — 未検出 ❌ "
                            f"(累計: 成功{self.success_count}/失敗{self.fail_count})"
                        )
                        log.info("─" * 40)
                        continue
                    else:
                        self._retry_no_minigame_count = 0

                        # 検出できたので未検出連続をリセット
                        self._update_section_head_adjust(False)

                if not self.running:
                    break

                # ミニゲーム本体
                result = self._fishing_minigame()

                self.fish_count += 1
                if result:
                    self.success_count += 1
                    self._consecutive_fail = 0
                    tag = "成功 ✅"
                else:
                    self.fail_count += 1
                    self._consecutive_fail += 1
                    tag = "失敗 ❌"

                log.info(
                    f"[🎣 結果] 第 {self.fish_count} 回釣り — {tag} "
                    f"(累計: 成功{self.success_count}/失敗{self.fail_count})"
                )

                log.info("─" * 40)

                self.state = "次のループ待機"
                time.sleep(config.POST_CATCH_DELAY)

            except Exception as e:
                log.error(f"実行中例外: {e}")
                if not config.IL_RECORD:
                    self.input.safe_release()
                time.sleep(2)

        if not config.IL_RECORD:
            self.input.safe_release()

        self.state = "停止済み"
        self._pd_close_recording()
        log.info("釣りスレッドを停止しました")

        try:
            cv2.destroyWindow("Debug Overlay")
        except Exception:
            pass