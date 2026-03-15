"""
YOLO 目标检测器
==============
封装 ultralytics YOLO 推理，提供与模板匹配 Detector 兼容的接口。

检测类别:
  0 = fish     (鱼图标)      → 返回 (x, y, w, h, conf)
  1 = bar      (白色捕捉条)  → 返回 (x, y, w, h, conf)
  2 = track    (钓鱼轨道)    → 返回 (x, y, w, h, conf)
  3 = progress_bar (进度条背景) → 返回 (x, y, w, h, conf)
  4 = progress (绿色进度条) → 返回 (x, y, w, h, conf)
"""

import os
import cv2
import numpy as np
from utils.logger import log

_YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    pass


class YoloDetector:
    """YOLO-based fishing game detector."""

    CLASS_FISH = 0
    CLASS_BAR = 1
    CLASS_TRACK = 2
    CLASS_PROGRESS_BAR = 3
    CLASS_PROGRESS = 4

    def __init__(self, model_path: str, conf: float = 0.5, device="auto"):
        if not _YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics 未安装。请运行: pip install ultralytics"
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO 模型未找到: {model_path}")

        self.conf = conf
        self.model = YOLO(model_path)

        import config as _cfg
        dev_pref = getattr(_cfg, "YOLO_DEVICE", "auto")
        cuda_ok = False
        try:
            import torch
            cuda_ok = torch.cuda.is_available()
        except Exception:
            pass
        if dev_pref == "cpu" or not cuda_ok:
            target_dev = "cpu"
        elif dev_pref == "gpu":
            target_dev = 0
        else:
            target_dev = 0

        warmup_img = np.zeros((640, 640, 3), dtype=np.uint8)

        if target_dev != "cpu":
            try:
                pass  # 静默加载
                self.model.predict(
                    warmup_img, conf=0.5, device=target_dev,
                    verbose=False, imgsz=640,
                )
                self._device = target_dev
                for _ in range(2):
                    self.model.predict(
                        warmup_img, conf=0.5, device=target_dev,
                        verbose=False, imgsz=640,
                    )
                pass  # GPU 预热完成
                return
            except Exception as e:
                if dev_pref == "gpu":
                    raise RuntimeError(f"[YOLO] 强制 GPU 模式但初始化失败: {e}")
                log.warning(f"[YOLO] GPU 不可用 ({e}), 回退 CPU")

        self._device = "cpu"
        pass  # 静默加载 CPU
        self.model.predict(
            warmup_img, conf=0.5, device="cpu",
            verbose=False, imgsz=640,
        )
        log.info(f"[YOLO] ✓ CPU 模式就绪: {self.model.names}")

    def detect_progress_fill_ratio(self, screen, progress_bar_box, progress_box):
        """
        progress_bar と progress の bbox から進捗率を返す
        戻り値: 0.0 ~ 1.0

        ルール:
        - progress_bar 未検出 → 0.0
        - progress_bar あり / progress 未検出 → 0.0
        - 両方あり → progress_bar 内での充填幅を割合化
        """
        if progress_bar_box is None:
            return 0.0

        bx, by, bw, bh = progress_bar_box[:4]
        if bw <= 1 or bh <= 1:
            return 0.0

        if progress_box is None:
            return 0.0

        px, py, pw, ph = progress_box[:4]
        if pw <= 1 or ph <= 1:
            return 0.0

        # bar の左右端
        bar_left = float(bx)
        bar_right = float(bx + bw)

        # progress の左右端
        prog_left = float(px)
        prog_right = float(px + pw)

        # progress を bar 範囲内にクリップ
        clipped_left = max(bar_left, prog_left)
        clipped_right = min(bar_right, prog_right)

        fill_w = max(0.0, clipped_right - clipped_left)
        ratio = fill_w / float(bw)

        return float(np.clip(ratio, 0.0, 1.0))

    def detect(self, screen, roi=None):
        ox, oy = 0, 0
        img = screen

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

        _h, _w = img.shape[:2]
        _max_dim = max(_h, _w)
        if _max_dim < 400:
            _infer_size = max(320, ((_max_dim + 31) // 32) * 32)
        else:
            _infer_size = 640

        results = self.model.predict(
            img, conf=self.conf, device=self._device,
            verbose=False, imgsz=_infer_size,
        )

        detections = {
            "fish": None,
            "bar": None,
            "track": None,
            "progress_bar": None,
            "progress": None,
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

            det = (bx, by, bw, bh, conf)
            class_name = self.model.names.get(cls, f"cls{cls}")
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
            elif class_name == "progress_bar":
                if detections["progress_bar"] is None or conf > detections["progress_bar"][4]:
                    detections["progress_bar"] = det
            elif class_name == "progress":
                if detections["progress"] is None or conf > detections["progress"][4]:
                    detections["progress"] = det

        return detections

    def detect_track(self, screen, roi=None):
        """仅检测轨道是否存在"""
        result = self.detect(screen, roi)
        return result["track"]

    def detect_bar(self, screen, roi=None):
        """仅检测白条"""
        result = self.detect(screen, roi)
        return result["bar"]

    def detect_fish(self, screen, roi=None):
        """仅检测鱼"""
        result = self.detect(screen, roi)
        return result["fish"], result["fish_name"]
