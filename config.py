"""
グローバル設定モジュール
========================
すべての調整可能パラメータをここで一元管理する。
"""

import os
import sys

# ═══════════════════════════════════════════════════════════
#  パス
# ═══════════════════════════════════════════════════════════
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "img")
_APP_DIR = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else BASE_DIR
DEBUG_DIR = os.path.join(_APP_DIR, "debug")
SETTINGS_FILE = os.path.join(_APP_DIR, "settings.json")

# ═══════════════════════════════════════════════════════════
#  VRChat ウィンドウ
# ═══════════════════════════════════════════════════════════
WINDOW_TITLE = "VRChat"

# ═══════════════════════════════════════════════════════════
#  ホットキー（VRChat内でも使用可能）
# ═══════════════════════════════════════════════════════════
HOTKEY_TOGGLE = "F9"
HOTKEY_STOP   = "F10"
HOTKEY_DEBUG  = "F11"

# ═══════════════════════════════════════════════════════════
#  時間パラメータ（秒）
# ═══════════════════════════════════════════════════════════
CAST_DELAY          = 0.3         # 投竿後の待機
BITE_TIMEOUT        = 60.0        # 最大待機時間（絶対上限）
BITE_FORCE_HOOK     = 0.3         # N秒間ヒットなし → 強制的に合わせてミニゲームへ（検出漏れ対策）
BITE_CHECK_INTERVAL = 0.15        # ヒット判定のチェック間隔
MIN_BITE_WAIT       = 0.2         # 最低N秒待ってからヒット検出開始（誤検出防止）
COLOR_BITE_WAIT     = 6.0         # N秒後に色検出を有効化（テンプレート優先）
COLOR_BITE_PIXELS   = 500         # 色検出の最小ピクセル数（高いほど厳しい）
HOOK_PRE_DELAY      = 0.1         # 合わせ前の遅延
HOOK_POST_DELAY     = 0.2         # 合わせ後 UI 出現待機
VERIFY_TIMEOUT      = 1.0         # ミニゲームUI出現確認のタイムアウト
VERIFY_CONSECUTIVE  = 1           # 連続Nフレーム検出で確定
GAME_LOOP_INTERVAL  = 0.005       # ミニゲームループ間隔（できるだけ高速）
SHOW_DEBUG             = False    # debugウィンドウ表示（OFFで高速化）
DEBUG_OVERLAY_INTERVAL = 0.033    # debugウィンドウ更新間隔
DEBUG_OVERLAY_MAX_W    = 1920     # debug最大幅
DEBUG_OVERLAY_MAX_H    = 1080     # debug最大高さ
TRACK_LOST_LIMIT    = 30          # Nフレーム魚＋バー消失 → ゲーム終了
FISH_LOST_LIMIT     = 60          # 魚がNフレーム消失 → 終了の可能性
SINGLE_OBJ_TIMEOUT  = 5.0         # 魚またはバー単独消失 N秒 → 失敗
OBJ_MIN_COUNT       = 1           # 1オブジェクト以上検出で継続
OBJ_GONE_LIMIT      = 80          # オブジェクト不足Nフレーム → 終了
POST_CATCH_DELAY    = 2.800       # 釣り終了後の待機
SHAKE_HEAD_TIME     = 0.0300      # 首振り時間
INITIAL_PRESS_TIME  = 0.2         # 開始時押下時間
SUCCESS_PROGRESS    = 0.42        # この進捗以上で成功判定
MINIGAME_TIMEOUT    = 480.0       # ミニゲーム最大時間
UI_CHECK_FRAMES     = 10          # NフレームごとにUI確認
UI_GONE_LIMIT       = 1           # UI消失N回で終了
SUCCESS_CLICK_DELAY = 0.3         # 成功後クリックまでの待機
POST_RELEASE_DELAY  = 0.3         # 竿を放してから次の行動までの待機 
ENABLE_SHAKE_HEAD = False
SHAKE_HEAD_FAIL_THRESHOLD = 3

ENABLE_SECTION_HEAD_ADJUST = True
HEAD_ADJUST_FAIL_THRESHOLD = 10
HEAD_ADJUST_SUCCESS_THRESHOLD = 3
HEAD_ADJUST_STEP_SEC = 0.1

# ═══════════════════════════════════════════════════════════
#  テンプレートマッチング信頼度
#  ★ ROI限定検索なので誤検出リスクが低い
# ═══════════════════════════════════════════════════════════
THRESH_BITE     = 0.50
THRESH_FISH     = 0.35
THRESH_BAR      = 0.40
THRESH_HOOK     = 0.45
THRESH_TRACK    = 0.35

# ═══════════════════════════════════════════════════════════
#  マルチスケールマッチング
# ═══════════════════════════════════════════════════════════
# 軌道検出スケール
MATCH_SCALES = [0.7, 1.0, 1.5, 2.0, 3.0]

# 白バー検出スケール
BAR_SCALES   = [0.7, 1.0, 1.5, 2.0, 3.0]

# ★ ゲーム内の魚アイコンの推定サイズ
#   テンプレートサイズ / FISH_GAME_SIZE から最適スケールを計算
FISH_GAME_SIZE = 30

# ═══════════════════════════════════════════════════════════
#  ミニゲーム制御
# ═══════════════════════════════════════════════════════════
# ── PD制御パラメータ（高慣性釣り向け）──
DEAD_ZONE       = 12
DEAD_ZONE_RATIO = 0.22
MAINTAIN_TAP_S  = 0.006
HOLD_MIN_S      = 0.015
HOLD_MAX_S      = 0.120
HOLD_GAIN       = 0.055
VELOCITY_SMOOTH = 0.3
PREDICT_AHEAD   = 0.22
SPEED_DAMPING   = 0.00025
MAX_FISH_BAR_DIST = 300
REGION_UP         = 300
REGION_DOWN       = 400
REGION_X          = 100
USE_OSC           = True
DETECT_ROI        = None

# ═══════════════════════════════════════════════════════════
#  安全保護
# ═══════════════════════════════════════════════════════════
PAUSE_ON_MOUSE_ZERO = False

# ═══════════════════════════════════════════════════════════
#  強制リセット
# ═══════════════════════════════════════════════════════════
ENABLE_FORCE_RESET = False
MAX_RETRY_NO_MINIGAME = 0
FORCE_RESET_DELAY = 15.0

# ═══════════════════════════════════════════════════════════
#  UI言語設定
# ═══════════════════════════════════════════════════════════
LANGUAGE = "jp"

# ═══════════════════════════════════════════════════════════
#  YOLO物体検出（テンプレートの代替）
# ═══════════════════════════════════════════════════════════
USE_YOLO      = True
YOLO_MODEL    = os.path.join(BASE_DIR, "yolo", "runs", "fish_detect", "weights", "best.pt")
YOLO_CONF     = 0.25
YOLO_DEVICE   = "auto"
YOLO_COLLECT  = True
YOLO_COLLECT_ON_FAIL = False
TRACK_MIN_ANGLE   = 3.0
TRACK_MAX_ANGLE   = 45.0

# ═══════════════════════════════════════════════════════════
#  行動模倣学習（Behavior Cloning）
# ═══════════════════════════════════════════════════════════
IL_RECORD       = False
IL_USE_MODEL    = False
IL_MODEL_PATH   = os.path.join(BASE_DIR, "imitation", "policy.pt")
IL_DATA_DIR     = os.path.join(BASE_DIR, "imitation", "data")
IL_HISTORY_LEN  = 10
IL_PRESS_THRESH = 0.50


# ── RL residual hold tuning ──
RL_ENABLE = True
RL_MODEL_PATH = os.path.join(BASE_DIR, "rl_hold_qtable.pkl")

RL_EPSILON = 0.25
RL_ALPHA = 0.05
RL_GAMMA = 0.96

# PD hold に対して加える補正候補（秒）
RL_HOLD_ACTIONS = [-0.1, -0.08, -0.05, -0.03, -0.015, -0.008, 0, 0.008, 0.015, 0.03, 0.05, 0.08, 0.1]

# 終端報酬
RL_SUCCESS_REWARD = 4.0
RL_FAIL_REWARD = -4.0

# ステップ報酬（v2）
RL_REWARD_ALIVE = 0.005
RL_REWARD_CENTER_SHAPE = 0.08
RL_REWARD_IN_BAR = 0.04
RL_REWARD_OUTSIDE_SCALE = 0.18
RL_REWARD_OUTSIDE_FLAT = 0.04
RL_REWARD_IMPROVE_GAIN = 0.015
RL_REWARD_RECAPTURE = 0.35
RL_REWARD_LOSE_BAR = 0.18
RL_REWARD_PROGRESS_GAIN = 0.60
RL_REWARD_ACTION_PENALTY = 0.35

# residual補正の暴れ防止
RL_MAX_DELTA_HOLD = 0.020

# 離散化
RL_ERR_BIN = 8
RL_VEL_BIN = 6
RL_FISHDELTA_BIN = 5
RL_FIB_BIN = 12
RL_HOLD_BIN = 6
RL_PROG_BIN = 6

RL_REPLAY_SIZE = 5000
RL_REPLAY_BATCH = 8
RL_REPLAY_WARMUP = 200

RL_LOG_STEP_REWARD = True
RL_LOG_STEP_INTERVAL = 1

PD_RECORD = False
PD_DATA_DIR = os.path.join(BASE_DIR, "imitation", "data")

# ═══════════════════════════════════════════════════════════
#  模板文件映射
# ═══════════════════════════════════════════════════════════
TEMPLATE_FILES = {
    "track":        "finshblock.png",
    "bar":          "block.png",
    "fish_white":   "wFish.png",
    "fish_green":   "greenFish.png",
    "fish_golden":  "goldenFish.png",
    "fish_copper":  "copperFish.png",
    "fish_blue":    "blueFish.png",
    "fish_purple":  "purpleFish.png",
    "fish_black":   "blackFish.png",
    "hook":         "gou.png",
    "prog_full":    "full.png",
    "prog_empty":   "null.png",
}

# 所有鱼模板 key 列表（find_fish 使用）
FISH_KEYS = [
    "fish_white", "fish_green", "fish_golden",
    "fish_copper", "fish_blue", "fish_purple", "fish_black",
    "fish_pink", "fish_red", "fish_rainbow",
]

# ═══════════════════════════════════════════════════════════
#  钓鱼白名单 (True=要钓, False=放弃)
# ═══════════════════════════════════════════════════════════
FISH_WHITELIST = {
    "fish_black":   True,   # 黑鱼
    "fish_white":   True,   # 白鱼
    "fish_copper":  True,   # 铜鱼
    "fish_green":   True,   # 绿鱼
    "fish_blue":    True,   # 蓝鱼
    "fish_purple":  True,   # 紫鱼
    "fish_pink":    True,   # 粉鱼
    "fish_red":     True,   # 红鱼
    "fish_rainbow": True,   # 彩鱼
}
