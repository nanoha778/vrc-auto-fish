"""
YOLO 物体検出器
==============
ultralytics YOLO 推論をラップし、
テンプレートマッチング Detector と互換性のあるインターフェースを提供する。

このクラスは「釣りミニゲームUIの検出」を担当する。

検出クラス:
  0 = fish      (魚アイコン)        → (x, y, w, h, conf)
  1 = bar       (白い捕捉バー)      → (x, y, w, h, conf)
  2 = track     (釣りミニゲームの軌道) → (x, y, w, h, conf)
  3 = progress  (緑の進捗バー)      → (x, y, w, h, conf)
  4 = hook      (進捗バー上端のフック位置) → (x, y, w, h, conf)

このモジュールの役割
----------------
YOLOモデルを使用して以下を検出する

・魚の位置
・白いキャッチバー
・釣りミニゲームの軌道
・進捗バー
・フック位置

そして bot.py が扱いやすい形式に変換する。
"""

import os
import cv2
import numpy as np
from utils.logger import log


# YOLOが利用可能か確認
_YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    pass


class YoloDetector:
    """YOLOを使った釣りゲーム検出クラス"""

    # クラスID定義（学習時のラベルID）
    CLASS_FISH = 0
    CLASS_BAR = 1
    CLASS_TRACK = 2
    CLASS_PROGRESS = 3
    CLASS_HOOK = 4

    def __init__(self, model_path: str, conf: float = 0.5, device="auto"):
        """
        YOLO検出器初期化

        Parameters
        ----------
        model_path : str
            YOLOモデルのパス

        conf : float
            検出信頼度閾値

        device : str
            使用デバイス
            auto / cpu / gpu
        """

        # YOLOライブラリ確認
        if not _YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics がインストールされていません。 pip install ultralytics を実行してください"
            )

        # モデル存在確認
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO モデルが見つかりません: {model_path}")

        self.conf = conf
        self.model = YOLO(model_path)

        # configからデバイス設定取得
        import config as _cfg
        dev_pref = getattr(_cfg, "YOLO_DEVICE", "auto")

        cuda_ok = False
        try:
            import torch
            cuda_ok = torch.cuda.is_available()
        except Exception:
            pass

        # 使用デバイス決定
        if dev_pref == "cpu" or not cuda_ok:
            target_dev = "cpu"
        elif dev_pref == "gpu":
            target_dev = 0
        else:
            target_dev = 0

        # GPU初期化ウォームアップ用画像
        warmup_img = np.zeros((640, 640, 3), dtype=np.uint8)

        # GPU使用
        if target_dev != "cpu":
            try:
                self.model.predict(
                    warmup_img,
                    conf=0.5,
                    device=target_dev,
                    verbose=False,
                    imgsz=640,
                )

                self._device = target_dev

                # CUDA安定化のため追加ウォームアップ
                for _ in range(2):
                    self.model.predict(
                        warmup_img,
                        conf=0.5,
                        device=target_dev,
                        verbose=False,
                        imgsz=640,
                    )

                return

            except Exception as e:
                if dev_pref == "gpu":
                    raise RuntimeError(f"[YOLO] GPU 強制モードだが初期化失敗: {e}")

                log.warning(f"[YOLO] GPUが使用できません ({e}) → CPUにフォールバック")

        # CPUモード
        self._device = "cpu"

        self.model.predict(
            warmup_img,
            conf=0.5,
            device="cpu",
            verbose=False,
            imgsz=640,
        )

        log.info(f"[YOLO] ✓ CPUモード初期化完了: {self.model.names}")


    def detect_progress_fill_ratio(self, screen, progress_box):
        """
        progressバーの緑色充填率を計算する

        Parameters
        ----------
        screen : ndarray
            BGR画像

        progress_box : (x,y,w,h)

        Returns
        -------
        float
            0.0 ~ 1.0
        """

        if progress_box is None:
            return 0.0

        x, y, w, h = progress_box[:4]
        h_img, w_img = screen.shape[:2]

        # ROI範囲を画像内にクリップ
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = max(1, min(w, w_img - x))
        h = max(1, min(h, h_img - y))

        roi = screen[y:y+h, x:x+w]

        if roi.size == 0:
            return 0.0

        # progressバー枠の影響を避けるため内側を切り出す
        pad_x = max(1, int(w * 0.03))
        pad_y = max(1, int(h * 0.15))

        x1 = pad_x
        y1 = pad_y
        x2 = max(x1 + 1, w - pad_x)
        y2 = max(y1 + 1, h - pad_y)

        roi = roi[y1:y2, x1:x2]

        if roi.size == 0:
            return 0.0

        # HSV変換
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 緑色検出範囲
        lower = np.array([35, 60, 60], dtype=np.uint8)
        upper = np.array([90, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)

        # 列方向の充填率
        col_ratio = mask.mean(axis=0) / 255.0

        # 充填されている列
        filled = col_ratio > 0.25

        if not np.any(filled):
            return 0.0

        # 一番右の充填列
        rightmost = np.where(filled)[0].max()

        fill_ratio = (rightmost + 1) / len(filled)

        return float(np.clip(fill_ratio, 0.0, 1.0))


    def detect_hook_progress(self, progress_box, hook_box):
        """
        progress下端 → hook中心までの距離を
        progress高さで正規化して返す (0~1)

        釣りミニゲームの進行位置推定に使用
        """

        if progress_box is None or hook_box is None:
            return None

        px, py, pw, ph = progress_box[:4]
        hx, hy, hw, hh = hook_box[:4]

        progress_bottom = py + ph
        hook_center_y = hy + hh * 0.5

        distance = progress_bottom - hook_center_y

        if distance < 0:
            distance = 0

        return float(np.clip(distance / ph, 0.0, 1.0))


    def detect(self, screen, roi=None):
        """
        1フレームに対してYOLO推論を実行

        Parameters
        ----------
        screen : ndarray
            BGR画像

        roi : [x,y,w,h]
            検出範囲（任意）

        Returns
        -------
        dict
        """

        ox, oy = 0, 0
        img = screen

        # ROIが指定されている場合
        if roi:
            rx, ry, rw, rh = roi
            h_s, w_s = screen.shape[:2]

            rx = max(0, min(rx, w_s))
            ry = max(0, min(ry, h_s))
            rw = min(rw, w_s - rx)
            rh = min(rh, h_s - ry)

            if rw > 10 and rh > 10:
                img = screen[ry:ry+rh, rx:rx+rw].copy()
                ox, oy = rx, ry

        # 推論解像度を自動調整
        _h, _w = img.shape[:2]
        _max_dim = max(_h, _w)

        if _max_dim < 400:
            _infer_size = max(320, ((_max_dim + 31) // 32) * 32)
        else:
            _infer_size = 640

        results = self.model.predict(
            img,
            conf=self.conf,
            device=self._device,
            verbose=False,
            imgsz=_infer_size,
        )

        detections = {
            "fish": None,
            "bar": None,
            "track": None,
            "progress": None,
            "hook": None,
            "fish_name": "",
            "raw": [],
        }

        if not results or len(results) == 0:
            return detections

        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            return detections

        for i in range(len(boxes)):

            cls = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()

            bx = int(x1) + ox
            by = int(y1) + oy
            bw = int(x2 - x1)
            bh = int(y2 - y1)

            if bw <= 0 or bh <= 0:
                continue

            det = (bx, by, bw, bh, conf)

            if isinstance(self.model.names, dict):
                class_name = self.model.names.get(cls, f"cls{cls}")
            else:
                class_name = self.model.names[cls] if 0 <= cls < len(self.model.names) else f"cls{cls}"

            detections["raw"].append((class_name, det))

            if class_name == "fish":
                if detections["fish"] is None or conf > detections["fish"][4]:
                    detections["fish"] = det
                    detections["fish_name"] = "fish"

            elif class_name == "bar":
                if detections["bar"] is None or conf > detections["bar"][4]:
                    detections["bar"] = det

            elif class_name == "track":
                if detections["track"] is None or conf > detections["track"][4]:
                    detections["track"] = det

            elif class_name == "progress":
                if detections["progress"] is None or conf > detections["progress"][4]:
                    detections["progress"] = det

            elif class_name == "hook":
                if detections["hook"] is None or conf > detections["hook"][4]:
                    detections["hook"] = det

        return detections


    def detect_track(self, screen, roi=None):
        """軌道のみ検出"""
        result = self.detect(screen, roi)
        return result["track"]


    def detect_bar(self, screen, roi=None):
        """白バーのみ検出"""
        result = self.detect(screen, roi)
        return result["bar"]


    def detect_fish(self, screen, roi=None):
        """魚のみ検出"""
        result = self.detect(screen, roi)
        return result["fish"], result["fish_name"]


    def detect_progress(self, screen, roi=None):
        """進捗バーのみ検出"""
        result = self.detect(screen, roi)
        return result["progress"]


    def detect_hook(self, screen, roi=None):
        """フックのみ検出"""
        result = self.detect(screen, roi)
        return result["hook"]