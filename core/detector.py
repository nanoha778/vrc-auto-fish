"""
画像認識モジュール
================
OpenCV のテンプレートマッチング + 多スケール探索 + 色検出に基づく認識処理。

改良点:
- 多スケールマッチング:
    解像度や DPI スケーリング差に対応するため、
    複数倍率を自動で試す
- グレースケールマッチング:
    色差の影響を減らす
- デバッグレポート:
    debug モードでは、閾値未満でも最高信頼度を表示する
"""

import cv2
import numpy as np
import os

import config
from utils.logger import log


class ImageDetector:
    """画像検出器"""

    def __init__(self, img_dir: str, template_files: dict):
        # カラーテンプレート
        self.templates = {}
        # グレースケールテンプレート
        self.templates_gray = {}

        self.debug_report = False          # bot 側から設定される
        self._last_scale = 1.0             # 直近の find_multiscale 命中スケール
        self._last_best_key = None         # 直近の find_best で最良だった key
        self._last_best_scale = 1.0        # 直近の find_best で最良だった scale

        # CUDA 関連
        self._use_cuda = False
        self._cuda_matcher = None

        # リサイズ済みテンプレートのキャッシュ
        self._scaled_cache = {}
        self._gpu_scaled_cache = {}

        self._load_templates(img_dir, template_files)
        self._init_gpu()

    # ══════════════════ テンプレート読み込み ══════════════════

    _TMPL_MAX_DIM = 9999
    # テンプレートを実質トリミングしない設定
    # 完全なテンプレートを保持して精度を優先する

    _SCALE_CACHE_MAX = 200
    # リサイズ済みテンプレートのキャッシュ上限
    # 超えたら古い半分を捨てる（メモリ / VRAM増加抑制）

    def _load_templates(self, img_dir: str, file_map: dict):
        """
        テンプレート画像を読み込む。

        Parameters
        ----------
        img_dir : str
            テンプレート画像フォルダ

        file_map : dict
            {論理名: ファイル名}
        """
        pass  # 静かに読み込む（通常ログを出しすぎない）

        for key, fname in file_map.items():
            path = os.path.join(img_dir, fname)
            img = cv2.imread(path, cv2.IMREAD_COLOR)

            if img is not None:
                h, w = img.shape[:2]
                orig_desc = f"{w}×{h}"

                # 異常に縦長 / 横長のテンプレートが来た場合のみ中心切り抜き
                mx = self._TMPL_MAX_DIM

                if h > mx:
                    cy = h // 2
                    y0 = cy - mx // 2
                    img = img[y0:y0 + mx, :, :]
                    h = mx

                if w > mx:
                    cx = w // 2
                    x0 = cx - mx // 2
                    img = img[:, x0:x0 + mx, :]
                    w = mx

                if orig_desc != f"{w}×{h}":
                    pass
                else:
                    pass

                self.templates[key] = img
                self.templates_gray[key] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            else:
                self.templates[key] = None
                self.templates_gray[key] = None
                log.warning(f"  ✗ {fname:<15s}  (未找到)")

    # ══════════════════ GPU / CUDA ══════════════════

    def _init_gpu(self):
        """
        CUDA 加速を有効化してみる。
        利用できなければ CPU にフォールバックする。
        """
        self._use_cuda = False
        self._cuda_matcher = None
        self._gpu_templates = {}

        # OpenCL は使わず、CUDA 専用にする
        cv2.ocl.setUseOpenCL(False)

        try:
            if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self._cuda_matcher = cv2.cuda.createTemplateMatching(
                    cv2.CV_8U, cv2.TM_CCOEFF_NORMED
                )

                # 動作確認用の小さなダミー画像でマッチャ初期化
                _t = np.zeros((32, 32), dtype=np.uint8)
                _s = np.zeros((8, 8), dtype=np.uint8)
                self._cuda_matcher.match(
                    cv2.cuda_GpuMat(_t), cv2.cuda_GpuMat(_s)
                )

                self._use_cuda = True

                # グレースケールテンプレートを GPU に載せておく
                for key, tmpl in self.templates_gray.items():
                    if tmpl is not None:
                        self._gpu_templates[key] = cv2.cuda_GpuMat(tmpl)

                dev = cv2.cuda.getDevice()
                dev_info = cv2.cuda.DeviceInfo(dev)
                vram_mb = dev_info.totalMemory() // 1048576

                log.info(f"[エンジン] ✓ CUDA 有効: GPU #{dev} ({vram_mb} MB)")
                log.info(
                    f"  GPUテンプレートキャッシュ: {len(self._gpu_templates)} 個"
                )
                return

        except Exception as e:
            self._use_cuda = False
            self._cuda_matcher = None
            self._gpu_templates = {}
            log.debug(f"[エンジン] CUDA 初期化失敗: {e}")

        pass  # 静かに CPU モードへ

    _CUDA_MIN_PIXELS = 50_000
    # 画像が小さいと GPU 転送のほうが遅くなるので、
    # 一定以上の面積のときだけ CUDA を使う

    def _match_template(self, img_gray, tmpl_gray):
        """
        テンプレートマッチング本体（CPU経路）
        """
        result = cv2.matchTemplate(img_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        return max_val, max_loc

    def _cuda_match(self, gpu_img, gpu_tmpl):
        """
        テンプレートマッチング本体（CUDA経路）
        引数はすでに GPU 上にあるものを渡す。
        """
        gpu_result = self._cuda_matcher.match(gpu_img, gpu_tmpl)
        result_cpu = gpu_result.download()
        _, max_val, _, max_loc = cv2.minMaxLoc(result_cpu)
        return max_val, max_loc

    def _should_use_cuda(self, h, w):
        """
        CUDAを使うべきか判定する。
        一定以上の画像サイズのときだけ GPU を使う。
        """
        return self._use_cuda and h * w >= self._CUDA_MIN_PIXELS

    # ══════════════════ グレースケール準備 / キャッシュ ══════════════════

    _gray_cache_id = -1
    _gray_cache_img = None

    def prepare_gray(self, screen, search_region=None, upload_gpu=False):
        """
        探索領域のグレースケール画像を事前計算する。
        同一フレーム内で複数回 find_multiscale を呼ぶときに再利用できる。

        Parameters
        ----------
        screen : ndarray
            BGR画像

        search_region : tuple or None
            (x, y, w, h)

        upload_gpu : bool
            True のとき、可能なら GpuMat を返す

        Returns
        -------
        (gray_img_or_GpuMat, ox, oy)
            ox, oy は元画像に対するオフセット
        """
        ox, oy = 0, 0
        img = screen

        if search_region:
            rx, ry, rw, rh = (
                int(search_region[0]), int(search_region[1]),
                int(search_region[2]), int(search_region[3])
            )
            ox, oy = rx, ry
            img = screen[ry:ry + rh, rx:rx + rw]

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        if upload_gpu and self._should_use_cuda(*gray.shape[:2]):
            try:
                gray = np.ascontiguousarray(gray)
                return cv2.cuda_GpuMat(gray), ox, oy
            except Exception:
                pass

        return gray, ox, oy

    # ══════════════════ 単一スケールマッチング ══════════════════

    def find(self, screen, tmpl_key: str, threshold: float = 0.6,
             search_region=None):
        """
        単一スケール（1:1）テンプレートマッチング。

        Returns
        -------
        (x, y, w, h, confidence) または None
        """
        tmpl = self.templates.get(tmpl_key)
        if tmpl is None:
            return None

        ox, oy = 0, 0
        img = screen

        if search_region:
            rx, ry, rw, rh = [int(v) for v in search_region]
            ox, oy = rx, ry
            img = screen[ry: ry + rh, rx: rx + rw]

        th, tw = tmpl.shape[:2]
        if img.shape[0] < th or img.shape[1] < tw:
            return None

        if len(img.shape) == 3:
            img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_g = img

        tmpl_g = self.templates_gray.get(tmpl_key)
        if tmpl_g is None:
            tmpl_g = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY) if len(tmpl.shape) == 3 else tmpl

        if self._should_use_cuda(*img_g.shape[:2]):
            gpu_tmpl = self._gpu_templates.get(tmpl_key)
            if gpu_tmpl is not None:
                try:
                    max_val, max_loc = self._cuda_match(
                        cv2.cuda_GpuMat(np.ascontiguousarray(img_g)),
                        gpu_tmpl
                    )
                except Exception:
                    max_val, max_loc = self._match_template(img_g, tmpl_g)
            else:
                max_val, max_loc = self._match_template(img_g, tmpl_g)
        else:
            max_val, max_loc = self._match_template(img_g, tmpl_g)

        if max_val >= threshold:
            return (max_loc[0] + ox, max_loc[1] + oy, tw, th, max_val)

        if self.debug_report:
            log.debug(f"  {tmpl_key}: 最高信頼度 {max_val:.3f} (閾値 {threshold})")

        return None

    # ══════════════════ 多スケールマッチング ══════════════════

    def find_multiscale(self, screen, tmpl_key: str, threshold: float = 0.6,
                        search_region=None, scales=None,
                        pre_gray=None, pre_offset=None):
        """
        多スケールテンプレートマッチング。

        Parameters
        ----------
        pre_gray / pre_offset :
            prepare_gray() で事前計算したグレースケール画像とオフセット。
            渡された場合は切り抜き + グレースケール化をスキップする。
            同一フレームでの複数回呼び出しを高速化できる。

        pre_gray は numpy.ndarray または cv2.cuda.GpuMat を受け付ける。
        """
        tmpl = self.templates_gray.get(tmpl_key)
        if tmpl is None:
            return None

        if scales is None:
            scales = config.MATCH_SCALES

        # ── グレースケール画像準備 ──
        gpu_img = None

        if pre_gray is not None:
            if self._use_cuda and isinstance(pre_gray, cv2.cuda.GpuMat):
                gpu_img = pre_gray
                ih, iw = gpu_img.size()[1], gpu_img.size()[0]
            else:
                img_gray = pre_gray
            ox, oy = pre_offset or (0, 0)

        else:
            ox, oy = 0, 0
            img = screen

            if search_region:
                rx, ry, rw, rh = (
                    int(search_region[0]), int(search_region[1]),
                    int(search_region[2]), int(search_region[3])
                )
                ox, oy = rx, ry
                img = screen[ry:ry + rh, rx:rx + rw]

            if len(img.shape) == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img

        if gpu_img is None:
            ih, iw = img_gray.shape[:2]
            if self._should_use_cuda(ih, iw):
                try:
                    gpu_img = cv2.cuda_GpuMat(
                        np.ascontiguousarray(img_gray)
                    )
                except Exception:
                    pass

        th, tw = tmpl.shape[:2]
        best_val = 0.0
        best_match = None
        best_scale = 1.0

        # ════════════ CUDA経路 ════════════
        # 画像がすでに GPU 上にあり、テンプレートキャッシュも使える
        if gpu_img is not None:
            gpu_tmpl_orig = self._gpu_templates.get(tmpl_key)

            for scale in scales:
                try:
                    if scale == 1.0:
                        if ih < th or iw < tw or gpu_tmpl_orig is None:
                            continue

                        max_val, max_loc = self._cuda_match(
                            gpu_img, gpu_tmpl_orig
                        )

                        if max_val > best_val:
                            best_val = max_val
                            best_scale = scale
                            if max_val >= threshold:
                                best_match = (
                                    max_loc[0] + ox, max_loc[1] + oy,
                                    tw, th, max_val
                                )

                    elif scale < 1.0:
                        # 画像を縮小して原寸テンプレートで探す
                        new_w = int(iw * scale)
                        new_h = int(ih * scale)
                        if new_w < tw or new_h < th:
                            continue

                        gpu_img_s = cv2.cuda.resize(
                            gpu_img, (new_w, new_h),
                            interpolation=cv2.INTER_AREA
                        )

                        if gpu_tmpl_orig is None:
                            continue

                        max_val, max_loc = self._cuda_match(
                            gpu_img_s, gpu_tmpl_orig
                        )

                        if max_val > best_val:
                            best_val = max_val
                            best_scale = scale
                            if max_val >= threshold:
                                real_x = int(max_loc[0] / scale) + ox
                                real_y = int(max_loc[1] / scale) + oy
                                best_match = (
                                    real_x, real_y,
                                    int(tw / scale), int(th / scale),
                                    max_val
                                )

                    else:  # scale > 1.0
                        # テンプレートを縮小して原寸画像で探す
                        new_tw = int(tw / scale)
                        new_th = int(th / scale)

                        if new_tw < 15 or new_th < 15:
                            continue
                        if ih < new_th or iw < new_tw:
                            continue

                        _gkey = (tmpl_key, new_tw, new_th)
                        gpu_tmpl_s = self._gpu_scaled_cache.get(_gkey)

                        if gpu_tmpl_s is None:
                            scaled_tmpl = cv2.resize(
                                tmpl, (new_tw, new_th),
                                interpolation=cv2.INTER_LINEAR
                            )
                            scaled_tmpl = np.ascontiguousarray(scaled_tmpl)

                            try:
                                gpu_tmpl_s = cv2.cuda_GpuMat(scaled_tmpl)

                                if len(self._gpu_scaled_cache) >= self._SCALE_CACHE_MAX:
                                    # 古い半分を捨てて最近使うものを残す
                                    _keys = list(self._gpu_scaled_cache)
                                    for _k in _keys[:len(_keys) // 2]:
                                        del self._gpu_scaled_cache[_k]

                                self._gpu_scaled_cache[_gkey] = gpu_tmpl_s

                            except Exception:
                                # GPU テンプレート作成失敗時は CPU でフォールバック
                                max_val, max_loc = self._match_template(
                                    img_gray if gpu_img is None else gpu_img.download(),
                                    scaled_tmpl
                                )
                                if max_val > best_val:
                                    best_val = max_val
                                    best_scale = scale
                                    if max_val >= threshold:
                                        best_match = (
                                            max_loc[0] + ox, max_loc[1] + oy,
                                            new_tw, new_th, max_val
                                        )
                                continue

                        max_val, max_loc = self._cuda_match(
                            gpu_img, gpu_tmpl_s
                        )

                        if max_val > best_val:
                            best_val = max_val
                            best_scale = scale
                            if max_val >= threshold:
                                best_match = (
                                    max_loc[0] + ox, max_loc[1] + oy,
                                    new_tw, new_th, max_val
                                )

                except cv2.error:
                    continue

        # ════════════ CPU経路 ════════════
        else:
            for scale in scales:
                if scale == 1.0:
                    if ih < th or iw < tw:
                        continue

                    max_val, max_loc = self._match_template(img_gray, tmpl)

                    if max_val > best_val:
                        best_val = max_val
                        best_scale = scale
                        if max_val >= threshold:
                            best_match = (
                                max_loc[0] + ox, max_loc[1] + oy,
                                tw, th, max_val
                            )

                elif scale < 1.0:
                    # 画像縮小
                    new_w = int(iw * scale)
                    new_h = int(ih * scale)
                    if new_w < tw or new_h < th:
                        continue

                    scaled_img = cv2.resize(
                        img_gray, (new_w, new_h),
                        interpolation=cv2.INTER_AREA
                    )
                    max_val, max_loc = self._match_template(scaled_img, tmpl)

                    if max_val > best_val:
                        best_val = max_val
                        best_scale = scale
                        if max_val >= threshold:
                            real_x = int(max_loc[0] / scale) + ox
                            real_y = int(max_loc[1] / scale) + oy
                            best_match = (
                                real_x, real_y,
                                int(tw / scale), int(th / scale),
                                max_val
                            )

                else:
                    # テンプレート縮小
                    new_tw = int(tw / scale)
                    new_th = int(th / scale)

                    if new_tw < 15 or new_th < 15:
                        continue
                    if ih < new_th or iw < new_tw:
                        continue

                    _ckey = (tmpl_key, new_tw, new_th)
                    scaled_tmpl = self._scaled_cache.get(_ckey)

                    if scaled_tmpl is None:
                        scaled_tmpl = cv2.resize(
                            tmpl, (new_tw, new_th),
                            interpolation=cv2.INTER_LINEAR
                        )

                        if len(self._scaled_cache) >= self._SCALE_CACHE_MAX:
                            # 古い半分を破棄
                            _keys = list(self._scaled_cache)
                            for _k in _keys[:len(_keys) // 2]:
                                del self._scaled_cache[_k]

                        self._scaled_cache[_ckey] = scaled_tmpl

                    max_val, max_loc = self._match_template(
                        img_gray, scaled_tmpl
                    )

                    if max_val > best_val:
                        best_val = max_val
                        best_scale = scale
                        if max_val >= threshold:
                            best_match = (
                                max_loc[0] + ox, max_loc[1] + oy,
                                new_tw, new_th, max_val
                            )

        self._last_scale = best_scale

        if self.debug_report:
            if best_match:
                log.debug(
                    f"  {tmpl_key}(多スケール): ✓ 信頼度 {best_val:.3f} "
                    f"@ scale={best_scale:.2f} (閾値 {threshold})"
                )
            else:
                log.debug(
                    f"  {tmpl_key}(多スケール): ✗ 最高信頼度 {best_val:.3f} "
                    f"@ scale={best_scale:.2f} (閾値 {threshold})"
                )

        return best_match

    # ══════════════════ 色による食いつき検出 ══════════════════

    def detect_bite_by_color(self, screen, min_cluster: int = 400) -> bool:
        """
        色特徴で食いつきマーク（感嘆符）を検出する。

        感嘆符の特徴:
        - 鮮やかなシアン〜青の縁取りがある
          （HSV ≈ H:85-130, S:100+, V:150+）
        - 1つのまとまった大きな塊として現れる

        重要な改良点:
        - 単純な青色ピクセル総数では判定しない
          （夜景や背景の青ブロックが誤検出を起こす）
        - 代わりに「単一の大きな連結成分」があるかを見る
        """
        h_scr, w_scr = screen.shape[:2]

        # 探索範囲は中央寄り
        # 感嘆符は竿 / ウキの近くに出る想定
        x1 = int(w_scr * 0.25)
        x2 = int(w_scr * 0.75)
        y1 = int(h_scr * 0.05)
        y2 = int(h_scr * 0.65)
        roi = screen[y1:y2, x1:x2]

        if roi.size == 0:
            return False

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 感嘆符の典型的なシアン〜青色
        mask = cv2.inRange(
            hsv,
            np.array([85, 100, 150]),
            np.array([130, 255, 255])
        )

        # 形態学処理:
        # 小ノイズを除去し、近いピクセルを連結しやすくする
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 最大連結成分のみを見る
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        largest_area = 0
        largest_contour = None
        for c in contours:
            area = cv2.contourArea(c)
            if area > largest_area:
                largest_area = area
                largest_contour = c

        detected = largest_area >= min_cluster

        # 追加条件:
        # 感嘆符はおおむね縦長（高さ > 幅）
        if detected and largest_contour is not None:
            _, _, cw, ch = cv2.boundingRect(largest_contour)
            if cw > 0 and ch > 0:
                aspect = ch / cw
                if aspect < 1.0:
                    detected = False
                    if self.debug_report:
                        log.debug(
                            f"  色検出(bite): 最大塊={largest_area} "
                            f"だが形状不一致(縦横比={aspect:.1f})、ブロックの可能性"
                        )

        total_px = int(cv2.countNonZero(mask))

        if self.debug_report:
            log.debug(
                f"  色検出(bite): 総ピクセル={total_px} "
                f"最大塊={largest_area} (閾値={min_cluster}) "
                f"→ {'✓ 検出' if detected else '✗'}"
            )

        # デバッグ用マスク保存
        if self.debug_report and detected:
            try:
                import config as _cfg
                path = os.path.join(_cfg.DEBUG_DIR, "bite_color_mask.png")
                cv2.imwrite(path, mask)
            except Exception:
                pass

        return detected

    # ══════════════════ 複合検出ユーティリティ ══════════════════

    def find_best(self, screen, keys: list, thresholds: list,
                  search_region=None, multiscale: bool = False):
        """
        複数テンプレートを試し、最も信頼度の高いマッチを返す。
        同時に最良 key / scale も記録する。

        ★ 最初に閾値超えしたものが出たら即返す
          （同一シーンに魚種は1種類しかない想定）
        """
        self._last_best_key = None
        self._last_best_scale = 1.0

        best = None
        best_conf = 0.0

        for key, thr in zip(keys, thresholds):
            if multiscale:
                m = self.find_multiscale(screen, key, thr, search_region)
            else:
                m = self.find(screen, key, thr, search_region)

            if m and m[4] > best_conf:
                best = m
                best_conf = m[4]
                self._last_best_key = key
                self._last_best_scale = self._last_scale
                break

        return best

    def _fish_scales_for(self, tmpl_key: str) -> list:
        """
        config.FISH_GAME_SIZE に基づいて、
        魚テンプレートの推奨探索スケールを自動生成する。

        原理:
            optimal_scale = テンプレートサイズ / ゲーム内魚サイズ

        例:
            テンプレート38px, ゲーム内魚20px → optimal=1.9
        """
        tmpl = self.templates.get(tmpl_key)
        game_size = getattr(config, 'FISH_GAME_SIZE', 0)

        if tmpl is None or game_size <= 0:
            return config.MATCH_SCALES

        h, w = tmpl.shape[:2]
        tmpl_size = max(h, w)
        optimal = tmpl_size / game_size

        scales = sorted(set(
            round(optimal * f, 2)
            for f in [0.6, 0.8, 1.0, 1.25, 1.5]
        ))

        # 下限 / 上限を制限
        scales = [s for s in scales if 0.3 <= s <= 5.0]

        return scales if scales else config.MATCH_SCALES

    def find_fish(self, screen, threshold: float, search_region=None,
                  pre_gray=None, pre_offset=None, keys=None):
        """
        魚アイコンを探す。
        複数の魚テンプレートを走査し、最も信頼度の高いものを返す。

        Parameters
        ----------
        keys : list or None
            指定された魚テンプレートだけ探索する
            （フレーム分割探索で高速化する用途）
        """
        self._last_best_key = None
        self._last_best_scale = 1.0

        best_match = None
        best_conf = 0.0
        best_key = None
        best_scale = 1.0

        for k in (keys or config.FISH_KEYS):
            # 白魚は誤検出しやすいので少し厳しめ
            thr = max(threshold, 0.75) if k == "fish_white" else threshold
            scales = self._fish_scales_for(k)

            m = self.find_multiscale(
                screen, k, thr, search_region, scales=scales,
                pre_gray=pre_gray, pre_offset=pre_offset,
            )

            if m and m[4] > best_conf:
                best_conf = m[4]
                best_match = m
                best_key = k
                best_scale = self._last_scale

        if best_match is not None:
            self._last_best_key = best_key
            self._last_best_scale = best_scale

        return best_match

    def identify_fish_type(self, screen, fish_box, debug_save=False):
        """
        YOLOで魚位置だけ取れた後、
        魚ボックス内部の中心領域の色から魚種を推定する。

        方法:
        - 枠の中心70%だけ使う（端の背景を除外）
        - 高彩度ピクセルだけ対象にする
        - 色相ヒストグラムのピークで代表色を決める
        """
        import os
        import numpy as np

        fx, fy, fw, fh = fish_box[:4]
        h_img, w_img = screen.shape[:2]

        # YOLO枠の中央 70% を使う
        mx = int(fw * 0.15)
        my = int(fh * 0.15)
        x1 = max(0, fx + mx)
        y1 = max(0, fy + my)
        x2 = min(w_img, fx + fw - mx)
        y2 = min(h_img, fy + fh - my)

        if x2 - x1 < 3 or y2 - y1 < 3:
            x1, y1 = max(0, fx), max(0, fy)
            x2, y2 = min(w_img, fx + fw), min(h_img, fy + fh)

        crop = screen[y1:y2, x1:x2]
        if crop.size == 0:
            return "fish_golden"

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_ch = hsv[:, :, 0].flatten()
        s_ch = hsv[:, :, 1].flatten()
        v_ch = hsv[:, :, 2].flatten()

        mask = (s_ch > 70) & (v_ch > 50)
        n_sat = int(mask.sum())

        # 彩度ピクセルが少なすぎる場合は明るさだけで白 / 黒を判定
        if n_sat < 5:
            v_mean = float(v_ch.mean())
            result = "fish_white" if v_mean > 130 else "fish_black"

        else:
            h_fish = h_ch[mask]
            red_count = int(np.sum((h_fish < 12) | (h_fish > 165)))
            h_dom = -1

            # 赤は色相の循環端にまたがるため特別扱い
            if red_count > n_sat * 0.35:
                result = "fish_red"
            else:
                hist, _ = np.histogram(h_fish, bins=18, range=(0, 180))
                peak = int(np.argmax(hist))
                h_dom = peak * 10 + 5

                if h_dom < 15 or h_dom > 165:
                    result = "fish_red"
                elif h_dom < 25:
                    result = "fish_copper"
                elif h_dom < 40:
                    result = "fish_golden"
                elif h_dom < 80:
                    result = "fish_green"
                elif h_dom < 115:
                    result = "fish_blue"
                elif h_dom < 140:
                    result = "fish_purple"
                elif h_dom < 165:
                    result = "fish_pink"
                else:
                    result = "fish_rainbow"

        if debug_save:
            full_crop = screen[max(0, fy):min(h_img, fy + fh),
                               max(0, fx):min(w_img, fx + fw)]
            dbg = full_crop.copy() if full_crop.size > 0 else crop.copy()

            info = f"{result} sat={n_sat} h={h_dom if n_sat >= 5 else -1}"
            cv2.putText(
                dbg, info, (2, dbg.shape[0] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1
            )

            debug_dir = os.path.join(config.BASE_DIR, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, "fish_id_crop.png"), dbg)

        return result

    def find_fish_by_color(self, screen, search_region=None,
                           bar_cx=None):
        """
        色ベースで魚位置を検出する。

        前提:
        - 魚は小さなピクセルスプライト
        - 鮮やかな高彩度色（緑 / 金 / 銅 / 青 / 紫 など）を持つ
        - 軌道背景上で目立つ

        Parameters
        ----------
        bar_cx : int or None
            白バー中心X
            分かっていれば軌道周辺に絞り込み、背景誤検出を大きく減らせる
        """
        if search_region is None:
            return None

        rx, ry, rw, rh = [int(v) for v in search_region]

        # 白バー位置が分かれば、その近辺だけを見る
        if bar_cx is not None:
            strip_half = 40
            new_rx = max(0, int(bar_cx) - strip_half)
            new_rw = min(strip_half * 2, screen.shape[1] - new_rx)

            # 元の探索範囲との共通部分だけ残す
            left = max(rx, new_rx)
            right = min(rx + rw, new_rx + new_rw)
            if right > left:
                rx, rw = left, right - left

        roi = screen[ry:ry + rh, rx:rx + rw]
        if roi.size == 0:
            return None

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 魚は高彩度 / 高明度寄り
        mask = cv2.inRange(
            hsv,
            np.array([0, 80, 80]),
            np.array([180, 255, 255])
        )

        # ノイズ除去
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 面積・形状でふるい分け
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if 50 < area < 4000:
                bx, by, bw, bh = cv2.boundingRect(c)
                aspect = max(bw, bh) / max(min(bw, bh), 1)
                if aspect < 3.0:
                    candidates.append((bx, by, bw, bh, area))

        if not candidates:
            if self.debug_report:
                total = int(cv2.countNonZero(mask))
                log.debug(f"  色魚検出: 高彩度ピクセル={total}, 有効輪郭なし")
            return None

        # 最大候補を魚とみなす
        candidates.sort(key=lambda c: c[4], reverse=True)
        bx, by, bw, bh, area = candidates[0]

        result = (rx + bx, ry + by, bw, bh, 0.55)

        if self.debug_report:
            log.debug(
                f"  色魚検出: ✓ 位置=({result[0]},{result[1]}) "
                f"サイズ={bw}×{bh} 面積={area}"
            )

        return result

    def find_catch_bar(self, screen, bar_thresh: float,
                       hook_thresh: float, search_region=None):
        """
        白い制御バーを探す。
        優先順位:
        1. テンプレートマッチング
        2. フック位置から補助推定
        3. （必要なら別色検出へ）

        ★ 白バーはゲーム内でテンプレより大きいことが多いので、
          BAR_SCALES（<= 1.0側中心）だけ使う
        """
        # 方法1: 白バーテンプレートを多スケールで探す
        bar = self.find_multiscale(
            screen, "bar", bar_thresh, search_region,
            scales=config.BAR_SCALES
        )
        if bar:
            return bar

        # 方法2: フック位置から白バー位置を補助推定
        hook = self.find_multiscale(screen, "hook", hook_thresh, search_region)
        if hook:
            bar_tmpl = self.templates.get("bar")
            bar_h = bar_tmpl.shape[0] if bar_tmpl is not None else 60
            return (hook[0], hook[1] - bar_h // 2, hook[2], bar_h, hook[4] * 0.9)

        return None

    def find_catch_bar_by_color(self, screen, strip_x: int, strip_w: int,
                                y_top: int, y_bottom: int):
        """
        色ベースで白バーを検出する（最後の保険手段）
        """
        x1 = max(0, strip_x)
        x2 = min(screen.shape[1], strip_x + strip_w)
        y1 = max(0, y_top)
        y2 = min(screen.shape[0], y_bottom)

        strip = screen[y1:y2, x1:x2]
        if strip.size == 0:
            return None

        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)

        # 低彩度・高明度 = 白っぽい帯
        mask = cv2.inRange(
            hsv,
            np.array([0, 0, 190]),
            np.array([180, 50, 255])
        )

        row_ratio = np.mean(mask > 0, axis=1)
        bright_rows = np.where(row_ratio > 0.3)[0]

        if len(bright_rows) < 5:
            return None

        center_y = y1 + int(np.mean(bright_rows))
        height = int(bright_rows[-1] - bright_rows[0])

        return (center_y, max(height, 10))

    # ══════════════════ 色による軌道検出（回転非依存） ══════════════════

    def detect_track_by_color(self, screen):
        """
        色ベースで釣り軌道を検出する（回転していても使える）。

        軌道の特徴:
        - 明るい青 / シアンの発光縁がある
        - 画面上で最も大きい細長い青領域
        - 内部に明るい白色領域（白ブロック）も含む

        Returns
        -------
        dict or None
            {
                'center': (cx, cy),
                'angle':  angle_deg,   # 0=垂直, 正=右傾き
                'length': long_side,
                'width':  short_side,
            }
        """
        hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
        h_scr, w_scr = screen.shape[:2]

        # 発光する青 / シアン縁
        blue_mask = cv2.inRange(
            hsv, np.array([85, 50, 100]), np.array([140, 255, 255])
        )

        # 内部白ブロック
        white_mask = cv2.inRange(
            hsv, np.array([0, 0, 200]), np.array([180, 40, 255])
        )

        combined = cv2.bitwise_or(blue_mask, white_mask)

        # 軌道内部の隙間を埋める + ノイズ除去
        kernel_close = np.ones((11, 11), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)

        kernel_open = np.ones((5, 5), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open)

        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 最も「大きくて細長い」輪郭を軌道候補にする
        best_contour = None
        best_score = 0
        min_length = max(h_scr, w_scr) * 0.15

        for c in contours:
            area = cv2.contourArea(c)
            if area < 1500:
                continue

            rect = cv2.minAreaRect(c)
            (_, _), (rw, rh), _ = rect
            long_side = max(rw, rh)
            short_side = max(min(rw, rh), 1)
            aspect = long_side / short_side

            if aspect < 3.5:
                continue
            if long_side < min_length:
                continue

            score = area * aspect
            if score > best_score:
                best_score = score
                best_contour = c

        if best_contour is None:
            if self.debug_report:
                log.debug("  色軌道検出: ✗ 細長い青領域が見つかりません")
            return None

        # fitLine で軸方向を高精度に求める
        vx, vy, x0, y0 = cv2.fitLine(
            best_contour, cv2.DIST_L2, 0, 0.01, 0.01
        )
        vx, vy = float(vx[0]), float(vy[0])

        # 上→下方向に揃える
        if vy < 0:
            vx, vy = -vx, -vy

        # 垂直方向からの偏角
        angle_deg = float(np.degrees(np.arctan2(vx, vy)))

        # [-90, 90] に正規化
        while angle_deg > 90:
            angle_deg -= 180
        while angle_deg < -90:
            angle_deg += 180

        rect = cv2.minAreaRect(best_contour)
        (cx, cy), (rw, rh), _ = rect

        result = {
            'center': (int(cx), int(cy)),
            'angle':  angle_deg,
            'length': float(max(rw, rh)),
            'width':  float(min(rw, rh)),
        }

        if self.debug_report:
            log.debug(
                f"  色軌道検出: ✓ 中心=({int(cx)},{int(cy)}) "
                f"角度={angle_deg:.1f}° 長={result['length']:.0f} "
                f"幅={result['width']:.0f}"
            )

        return result

    # ══════════════════ 進捗バー検出 ══════════════════

    def detect_green_ratio(self, screen, region) -> float:
        """
        指定領域内の緑色ピクセル比率を返す（進捗バー状態用）
        """
        x, y, w, h = [int(v) for v in region]
        x, y = max(0, x), max(0, y)
        w = min(w, screen.shape[1] - x)
        h = min(h, screen.shape[0] - y)

        if w <= 0 or h <= 0:
            return 0.0

        roi = screen[y: y + h, x: x + w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(
            hsv,
            np.array([35, 50, 50]),
            np.array([85, 255, 255])
        )

        total = mask.size
        return float(np.count_nonzero(mask)) / total if total > 0 else 0.0