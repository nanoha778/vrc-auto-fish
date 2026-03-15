"""
グローバル設定モジュール
================
すべての調整可能なパラメータをここで一括管理する。
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

CAST_DELAY = 1.0        # 釣り竿を投げた後の待機時間

BITE_TIMEOUT = 60.0     # 魚がかかる最大待機時間（絶対上限）

BITE_FORCE_HOOK = 0.500 # N秒間ヒットしない場合 → 強制的にフッキングしてミニゲームへ（検出漏れ対策）

BITE_CHECK_INTERVAL = 0.15 # ヒット検出の間隔

MIN_BITE_WAIT = 1.0        # 最低待機時間（誤検出防止）

COLOR_BITE_WAIT = 6.0      # N秒後にカラー検出を有効化（テンプレート優先）

COLOR_BITE_PIXELS = 500    # カラー検出の最低ピクセル数（高いほど厳しい）

HOOK_PRE_DELAY = 0.1       # フッキング前の遅延

HOOK_POST_DELAY = 0.4      # フッキング後 UI 出現待ち

VERIFY_TIMEOUT = 1.0       # ミニゲーム出現確認のタイムアウト

VERIFY_CONSECUTIVE = 1     # 連続Nフレーム白バー+トラック検出で確定

GAME_LOOP_INTERVAL = 0.005 # ミニゲームループ間隔

SHOW_DEBUG = True          # debugウィンドウ表示

DEBUG_OVERLAY_INTERVAL = 0.033

DEBUG_OVERLAY_MAX_W = 1920
DEBUG_OVERLAY_MAX_H = 1080

TRACK_LOST_LIMIT = 60
FISH_LOST_LIMIT = 120

SINGLE_OBJ_TIMEOUT = 5.0

OBJ_MIN_COUNT = 1

OBJ_GONE_LIMIT = 80

POST_CATCH_DELAY = 2.800

SHAKE_HEAD_TIME = 0.0300

INITIAL_PRESS_TIME = 0.2

SUCCESS_PROGRESS = 0.42

MINIGAME_TIMEOUT = 120.0

UI_CHECK_FRAMES = 5

UI_GONE_LIMIT = 2

SUCCESS_CLICK_DELAY = 0.2

POST_RELEASE_DELAY = 0.5


ENABLE_SECTION_HEAD_ADJUST = True
HEAD_ADJUST_FAIL_THRESHOLD = 10
HEAD_ADJUST_SUCCESS_THRESHOLD = 3
HEAD_ADJUST_STEP_SEC = 0.1


# ═══════════════════════════════════════════════════════════
# テンプレートマッチング閾値
# ═══════════════════════════════════════════════════════════

THRESH_BITE = 0.50

THRESH_FISH = 0.35
THRESH_BAR  = 0.40
THRESH_HOOK = 0.45
THRESH_TRACK = 0.35


# ═══════════════════════════════════════════════════════════
# 多スケールマッチング
# ═══════════════════════════════════════════════════════════

MATCH_SCALES = [0.7, 1.0, 1.5, 2.0, 3.0]

BAR_SCALES = [0.7, 1.0, 1.5, 2.0, 3.0]

FISH_GAME_SIZE = 30


# ═══════════════════════════════════════════════════════════
# ミニゲーム制御（PD制御）
# ═══════════════════════════════════════════════════════════

DEAD_ZONE = 12

DEAD_ZONE_RATIO = 0.22

MAINTAIN_TAP_S = 0.006

HOLD_MIN_S = 0.015

HOLD_MAX_S = 0.120

HOLD_GAIN = 0.055

VELOCITY_SMOOTH = 0.3

PREDICT_AHEAD = 0.22

SPEED_DAMPING = 0.00025

MAX_FISH_BAR_DIST = 300

REGION_UP = 300
REGION_DOWN = 400
REGION_X = 100

USE_OSC = True

DETECT_ROI = None


# ═══════════════════════════════════════════════════════════
# 安全保護機能
# ═══════════════════════════════════════════════════════════

PAUSE_ON_MOUSE_ZERO = False


# ═══════════════════════════════════════════════════════════
# 強制リセット機能
# ═══════════════════════════════════════════════════════════

ENABLE_FORCE_RESET = False

MAX_RETRY_NO_MINIGAME = 3

FORCE_RESET_DELAY = 15.0


# ═══════════════════════════════════════════════════════════
# UI言語設定
# ═══════════════════════════════════════════════════════════

LANGUAGE = "jp"


# ═══════════════════════════════════════════════════════════
# YOLO検出
# ═══════════════════════════════════════════════════════════

USE_YOLO = True

YOLO_MODEL = os.path.join(BASE_DIR, "yolo", "runs", "fish_detect", "weights", "best.pt")

YOLO_CONF = 0.25

YOLO_DEVICE = "auto"

YOLO_COLLECT = True

YOLO_COLLECT_ON_FAIL = False

TRACK_MIN_ANGLE = 3.0

TRACK_MAX_ANGLE = 45.0