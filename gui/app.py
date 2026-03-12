"""
tkinter GUI 模块
================
主控制面板：状态显示、控制按钮、参数调节、实时日志输出。
Bot 在后台线程运行，GUI 通过共享属性 + 日志队列通信。
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import queue
import json
import keyboard
import cv2

import config
from core.bot import FishingBot
from utils.logger import log
from utils import i18n


class FishingApp:
    """VRChat 自动钓鱼 — 主窗口"""

    def __init__(self, root: tk.Tk):
        self.root = root
        
        # ── 初始化语言 ──
        self._current_lang = getattr(config, 'LANGUAGE', 'zh')
        i18n.set_language(self._current_lang)
        
        self.root.title(i18n._("app_title"))
        self.root.geometry("580x820")
        self.root.resizable(True, True)
        self.root.minsize(520, 650)
        # ★ 默认不置顶 (用户可通过复选框开启)
        self.root.attributes("-topmost", False)

        # ── 机器人实例 ──
        self.bot = FishingBot()
        self.bot_thread: threading.Thread | None = None

        # ── 参数变量 ──
        self._param_vars = {}        # config属性名 → tk.StringVar

        # ── 构建界面 ──
        self._build_ui()

        # ── 加载上次保存的参数 ──
        self._load_settings()

        # ── 预加载 YOLO ──
        if self.bot.yolo is None:
            self._preload_yolo()

        # ── 注册全局快捷键 ──
        keyboard.add_hotkey(config.HOTKEY_TOGGLE, self._toggle_from_hotkey)
        keyboard.add_hotkey(config.HOTKEY_STOP,   self._stop_from_hotkey)
        keyboard.add_hotkey(config.HOTKEY_DEBUG,   self._toggle_debug_from_hotkey)

        # ── 启动轮询 ──
        self._poll()

        # ── 关闭处理 ──
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._log_msg(i18n._("msg_github"))

    # ══════════════════════════════════════════════════════
    #  界面构建
    # ══════════════════════════════════════════════════════

    def _build_ui(self):
        pad = {"padx": 10, "pady": 5}

        # ── 语言选择（最顶部）──
        frm_lang = ttk.Frame(self.root)
        frm_lang.pack(fill="x", padx=10, pady=(5, 0))
        ttk.Label(frm_lang, text=i18n._("language")).pack(side="left")
        self.var_language = tk.StringVar(value=self._current_lang)
        cmb_lang = ttk.Combobox(frm_lang, textvariable=self.var_language,
                                values=["zh", "en", "jp"],
                                state="readonly", width=10)
        cmb_lang.pack(side="left", padx=5)
        cmb_lang.bind("<<ComboboxSelected>>", self._on_language_change)

        # ── 顶部：状态面板 ──
        self.frm_status = ttk.LabelFrame(self.root, text=i18n._("status_group"))
        self.frm_status.pack(fill="x", **pad)

        grid_pad = {"padx": 8, "pady": 3, "sticky": "w"}

        self.var_state = tk.StringVar(value=i18n._("ready"))
        self.var_window = tk.StringVar(value=i18n._("not_connected"))
        self.var_count = tk.StringVar(value="0")
        self.var_debug = tk.StringVar(value=i18n._("debug_off"))  # 初始为关闭状态

        self.lbl_status_title = ttk.Label(self.frm_status, text=i18n._("running_status"))
        self.lbl_status_title.grid(row=0, column=0, **grid_pad)
        self.lbl_state = ttk.Label(self.frm_status, textvariable=self.var_state,
                                   foreground="gray")
        self.lbl_state.grid(row=0, column=1, **grid_pad)

        self.lbl_window_title = ttk.Label(self.frm_status, text=i18n._("window_status"))
        self.lbl_window_title.grid(row=1, column=0, **grid_pad)
        self.lbl_window = ttk.Label(self.frm_status, textvariable=self.var_window)
        self.lbl_window.grid(row=1, column=1, **grid_pad)

        self.lbl_count_title = ttk.Label(self.frm_status, text=i18n._("fish_count"))
        self.lbl_count_title.grid(row=2, column=0, **grid_pad)
        ttk.Label(self.frm_status, textvariable=self.var_count).grid(
            row=2, column=1, **grid_pad)

        # 成功率统计
        self.var_success_rate = tk.StringVar(value="0/0 (0%)")
        self.lbl_success_title = ttk.Label(self.frm_status, text=i18n._("success_rate"))
        self.lbl_success_title.grid(row=3, column=0, **grid_pad)
        ttk.Label(self.frm_status, textvariable=self.var_success_rate).grid(
            row=3, column=1, **grid_pad)

        # 强制重置统计
        self.var_force_reset_count = tk.StringVar(value="0")
        self.lbl_reset_title = ttk.Label(self.frm_status, text=i18n._("force_reset"))
        self.lbl_reset_title.grid(row=4, column=0, **grid_pad)
        frm_reset = ttk.Frame(self.frm_status)
        frm_reset.grid(row=4, column=1, **grid_pad)
        ttk.Label(frm_reset, textvariable=self.var_force_reset_count).pack(side="left")
        self.btn_view_log = ttk.Button(frm_reset, text=i18n._("btn_view_log"),
                   command=self._on_view_reset_log, width=8)
        self.btn_view_log.pack(side="left", padx=(5, 0))

        self.lbl_debug_title = ttk.Label(self.frm_status, text=i18n._("debug_mode"))
        self.lbl_debug_title.grid(row=5, column=0, **grid_pad)
        ttk.Label(self.frm_status, textvariable=self.var_debug).grid(
            row=5, column=1, **grid_pad)

        # ── 中间：控制按钮 ──
        frm_ctrl = ttk.Frame(self.root)
        frm_ctrl.pack(fill="x", **pad)

        self.btn_start = ttk.Button(frm_ctrl, text=i18n._("btn_start"),
                                    command=self._on_start, width=15)
        self.btn_start.pack(side="left", padx=5)

        self.btn_stop = ttk.Button(frm_ctrl, text=i18n._("btn_stop"),
                                   command=self._on_stop, width=15,
                                   state="disabled")
        self.btn_stop.pack(side="left", padx=5)

        self.btn_debug = ttk.Button(frm_ctrl, text=i18n._("btn_debug"),
                                    command=self._on_toggle_debug, width=15)
        self.btn_debug.pack(side="left", padx=5)

        # ── 辅助按钮 ──
        frm_aux = ttk.Frame(self.root)
        frm_aux.pack(fill="x", **pad)

        self.btn_connect = ttk.Button(frm_aux, text=i18n._("btn_connect"),
                                      command=self._on_connect, width=15)
        self.btn_connect.pack(side="left", padx=5)

        self.btn_screenshot = ttk.Button(frm_aux, text=i18n._("btn_screenshot"),
                                         command=self._on_screenshot, width=15)
        self.btn_screenshot.pack(side="left", padx=5)

        self.btn_clearlog = ttk.Button(frm_aux, text=i18n._("btn_clear_log"),
                                       command=self._on_clear_log, width=12)
        self.btn_clearlog.pack(side="left", padx=5)

        self.btn_whitelist = ttk.Button(frm_aux, text=i18n._("btn_whitelist"),
                                        command=self._on_whitelist, width=12)
        self.btn_whitelist.pack(side="left", padx=5)

        # ── 开关选项（独立一行，防窗口太窄时被挤掉）──
        frm_toggles = ttk.Frame(self.root)
        frm_toggles.pack(fill="x", **pad)

        self.var_topmost = tk.BooleanVar(value=False)
        self.chk_topmost = ttk.Checkbutton(frm_toggles, text=i18n._("opt_topmost"),
                        variable=self.var_topmost,
                        command=self._on_topmost)
        self.chk_topmost.pack(side="left", padx=5)

        self.var_show_debug = tk.BooleanVar(value=config.SHOW_DEBUG)
        self.chk_show_debug = ttk.Checkbutton(frm_toggles, text=i18n._("opt_debug_window"),
                        variable=self.var_show_debug,
                        command=self._on_debug_toggle)
        self.chk_show_debug.pack(side="left", padx=5)

        self.var_force_reset = tk.BooleanVar(value=config.ENABLE_FORCE_RESET)
        self.chk_force_reset = ttk.Checkbutton(frm_toggles, text=i18n._("opt_force_reset"),
                        variable=self.var_force_reset,
                        command=self._on_force_reset_toggle)
        self.chk_force_reset.pack(side="left", padx=5)

        # ── YOLO 控制区 ──
        self.frm_yolo = ttk.LabelFrame(self.root, text=i18n._("yolo_group"))
        self.frm_yolo.pack(fill="x", **pad)

        config.USE_YOLO = True
        self.lbl_yolo_enabled = ttk.Label(self.frm_yolo, text=i18n._("yolo_enabled"))
        self.lbl_yolo_enabled.pack(side="left", padx=5)

        self.var_yolo_collect = tk.BooleanVar(value=config.YOLO_COLLECT)
        self.chk_yolo_collect = ttk.Checkbutton(self.frm_yolo, text=i18n._("opt_collect_data"),
                        variable=self.var_yolo_collect,
                        command=self._on_yolo_collect_toggle)
        self.chk_yolo_collect.pack(side="left", padx=5)

        # 仅在失败时采集
        self.var_yolo_collect_on_fail = tk.BooleanVar(value=config.YOLO_COLLECT_ON_FAIL)
        self.chk_yolo_fail = ttk.Checkbutton(self.frm_yolo, text=i18n._("opt_collect_on_fail"),
                        variable=self.var_yolo_collect_on_fail,
                        command=self._on_yolo_collect_on_fail_toggle)
        self.chk_yolo_fail.pack(side="left", padx=5)

        self.lbl_yolo_device = ttk.Label(self.frm_yolo, text=i18n._("yolo_device"))
        self.lbl_yolo_device.pack(side="left", padx=(10, 2))
        self.var_yolo_device = tk.StringVar(value=config.YOLO_DEVICE)
        cmb_dev = ttk.Combobox(self.frm_yolo, textvariable=self.var_yolo_device,
                               values=["auto", "cpu", "gpu"],
                               state="readonly", width=5)
        cmb_dev.pack(side="left", padx=2)
        cmb_dev.bind("<<ComboboxSelected>>", self._on_yolo_device_change)

        self.var_yolo_status = tk.StringVar(value="")
        self._update_yolo_status()
        ttk.Label(self.frm_yolo, textvariable=self.var_yolo_status,
                  foreground="gray").pack(side="left", padx=10)

        # ── 检测区域框选 ──
        frm_roi = ttk.Frame(self.root)
        frm_roi.pack(fill="x", **pad)

        self.btn_roi = ttk.Button(frm_roi, text=i18n._("btn_select_roi"),
                                  command=self._on_select_roi, width=15)
        self.btn_roi.pack(side="left", padx=5)

        self.btn_clear_roi = ttk.Button(frm_roi, text=i18n._("btn_clear_roi"),
                                        command=self._on_clear_roi, width=12)
        self.btn_clear_roi.pack(side="left", padx=5)

        self.lbl_roi_title = ttk.Label(frm_roi, text=i18n._("detect_region"))
        self.lbl_roi_title.pack(side="left", padx=(10, 2))
        self.var_roi = tk.StringVar(value=i18n._("roi_not_set"))
        self.lbl_roi = ttk.Label(frm_roi, textvariable=self.var_roi,
                                 foreground="gray")
        self.lbl_roi.pack(side="left")

        # (行为克隆 UI 已移除)

        # ── 参数调节面板 ──
        self._build_params_panel(pad)

        # ── 底部：日志 ──
        self.frm_log = ttk.LabelFrame(self.root, text=i18n._("log_group"))
        self.frm_log.pack(fill="both", expand=True, **pad)

        self.txt_log = scrolledtext.ScrolledText(
            self.frm_log, height=14, state="disabled",
            font=("Consolas", 9), wrap="word",
            bg="#1e1e1e", fg="#d4d4d4",
            insertbackground="#d4d4d4",
        )
        self.txt_log.pack(fill="both", expand=True, padx=5, pady=5)

    # ══════════════════════════════════════════════════════
    #  语言切换
    # ══════════════════════════════════════════════════════

    def _on_language_change(self, _event=None):
        """切换界面语言"""
        lang = self.var_language.get()
        if lang == self._current_lang:
            return
        
        self._current_lang = lang
        i18n.set_language(lang)
        config.LANGUAGE = lang
        
        # 先应用当前参数值到config（保存用户输入）
        self._apply_params(save_only=True)
        
        # 更新所有界面文本
        self._update_ui_text()
        
        # 重建参数面板以更新标签语言
        self._rebuild_params_panel()
        
        # 保存设置
        self._save_settings()
        
        self._log_msg(f"[Language] Switched to {lang}")

    def _update_ui_text(self):
        """更新所有UI文本"""
        # 窗口标题
        self.root.title(i18n._("app_title"))
        
        # 状态面板
        self.frm_status.config(text=i18n._("status_group"))
        self.lbl_status_title.config(text=i18n._("running_status"))
        self.lbl_window_title.config(text=i18n._("window_status"))
        self.lbl_count_title.config(text=i18n._("fish_count"))
        self.lbl_success_title.config(text=i18n._("success_rate"))
        self.lbl_reset_title.config(text=i18n._("force_reset"))
        self.lbl_debug_title.config(text=i18n._("debug_mode"))
        self.btn_view_log.config(text=i18n._("btn_view_log"))
        
        # 更新状态变量（根据当前状态重新设置）
        if self.bot.running:
            self.var_state.set(i18n._("running"))
        else:
            # 根据bot.state判断是就绪还是已停止
            if self.bot.state in ["就绪", "Ready", "準備完了"]:
                self.var_state.set(i18n._("ready"))
            elif self.bot.state in ["已停止", "Stopped", "停止"]:
                self.var_state.set(i18n._("stopped"))
            else:
                self.var_state.set(i18n._("ready"))
        
        # 更新调试模式状态
        if self.bot.debug_mode:
            self.var_debug.set(i18n._("debug_on"))
        else:
            self.var_debug.set(i18n._("debug_off"))
        
        # 更新窗口连接状态
        current_window = self.var_window.get()
        if "HWND" in current_window or "(HWND=" in current_window:
            # 保持原有HWND信息，只更新前缀
            hwnd_start = current_window.find("(HWND=")
            if hwnd_start == -1:
                hwnd_start = current_window.find(" ")
            if hwnd_start > 0:
                hwnd_part = current_window[hwnd_start:]
                self.var_window.set(f"{i18n._('connected')} {hwnd_part}")
            else:
                self.var_window.set(i18n._("connected"))
        else:
            self.var_window.set(i18n._("not_connected"))
        
        # 按钮
        self.btn_start.config(text=i18n._("btn_start"))
        self.btn_stop.config(text=i18n._("btn_stop"))
        self.btn_debug.config(text=i18n._("btn_debug"))
        self.btn_connect.config(text=i18n._("btn_connect"))
        self.btn_screenshot.config(text=i18n._("btn_screenshot"))
        self.btn_clearlog.config(text=i18n._("btn_clear_log"))
        self.btn_whitelist.config(text=i18n._("btn_whitelist"))
        
        # 选项
        self.chk_topmost.config(text=i18n._("opt_topmost"))
        self.chk_show_debug.config(text=i18n._("opt_debug_window"))
        self.chk_force_reset.config(text=i18n._("opt_force_reset"))
        
        # YOLO
        self.frm_yolo.config(text=i18n._("yolo_group"))
        self.lbl_yolo_enabled.config(text=i18n._("yolo_enabled"))
        self.chk_yolo_collect.config(text=i18n._("opt_collect_data"))
        self.chk_yolo_fail.config(text=i18n._("opt_collect_on_fail"))
        self.lbl_yolo_device.config(text=i18n._("yolo_device"))
        
        # ROI
        self.btn_roi.config(text=i18n._("btn_select_roi"))
        self.btn_clear_roi.config(text=i18n._("btn_clear_roi"))
        self.lbl_roi_title.config(text=i18n._("detect_region"))
        if config.DETECT_ROI is None:
            self.var_roi.set(i18n._("roi_not_set"))
        
        # 日志
        self.frm_log.config(text=i18n._("log_group"))

    def _rebuild_params_panel(self):
        """重建参数面板以更新语言标签，保持原有位置"""
        # 销毁旧的面板
        self.frm_params.destroy()
        
        # 清空参数变量
        self._param_vars.clear()
        
        # 重新构建（不调用pack）
        pad = {"padx": 10, "pady": 5}
        self.frm_params = ttk.LabelFrame(self.root, text=i18n._("params_group"))
        
        # 使用 before 参数确保参数面板在日志区域之前
        self.frm_params.pack(fill="x", **pad, before=self.frm_log)
        
        # 4列布局: [标签 输入框] [标签 输入框]
        cols_per_row = 2
        gpad = {"padx": 4, "pady": 2}

        # 获取当前语言的参数列表
        tunable_params = i18n.get_tunable_params(self._current_lang)

        for i, (label, attr, vtype, tip) in enumerate(tunable_params):
            row = i // cols_per_row
            col_base = (i % cols_per_row) * 3   # 每组占3列: label, entry, unit

            # 从 config 读取当前值并转换为显示值
            display_val = self._config_to_display(attr, vtype)
            var = tk.StringVar(value=display_val)
            self._param_vars[attr] = (var, vtype)

            # 标签
            lbl = ttk.Label(self.frm_params, text=label, width=12, anchor="e")
            lbl.grid(row=row, column=col_base, sticky="e", **gpad)

            # 输入框
            entry = ttk.Entry(self.frm_params, textvariable=var, width=8,
                              justify="center")
            entry.grid(row=row, column=col_base + 1, sticky="w", **gpad)

            # 绑定回车和失焦自动应用
            entry.bind("<Return>", lambda e: self._apply_params())
            entry.bind("<FocusOut>", lambda e: self._apply_params())

            # 提示 (鼠标悬停)
            if tip:
                self._create_tooltip(entry, tip)

        # 按钮行
        total_rows = (len(tunable_params) + cols_per_row - 1) // cols_per_row
        btn_frame = ttk.Frame(self.frm_params)
        btn_frame.grid(row=total_rows, column=0, columnspan=6,
                       pady=(5, 5), sticky="e", padx=10)

        self.btn_apply = ttk.Button(btn_frame, text=i18n._("btn_apply"),
                   command=self._apply_params, width=10)
        self.btn_apply.pack(side="left", padx=3)
        self.btn_reset = ttk.Button(btn_frame, text=i18n._("btn_reset"),
                   command=self._reset_params, width=10)
        self.btn_reset.pack(side="left", padx=3)

    # ══════════════════════════════════════════════════════
    #  参数调节面板
    # ══════════════════════════════════════════════════════

    def _build_params_panel(self, pad):
        """构建小游戏参数实时调节面板"""
        self.frm_params = ttk.LabelFrame(self.root, text=i18n._("params_group"))
        self.frm_params.pack(fill="x", **pad)

        # 4列布局: [标签 输入框] [标签 输入框]
        cols_per_row = 2
        gpad = {"padx": 4, "pady": 2}

        # 获取当前语言的参数列表
        tunable_params = i18n.get_tunable_params(self._current_lang)

        for i, (label, attr, vtype, tip) in enumerate(tunable_params):
            row = i // cols_per_row
            col_base = (i % cols_per_row) * 3   # 每组占3列: label, entry, unit

            # 从 config 读取当前值并转换为显示值
            display_val = self._config_to_display(attr, vtype)
            var = tk.StringVar(value=display_val)
            self._param_vars[attr] = (var, vtype)

            # 标签
            lbl = ttk.Label(self.frm_params, text=label, width=12, anchor="e")
            lbl.grid(row=row, column=col_base, sticky="e", **gpad)

            # 输入框
            entry = ttk.Entry(self.frm_params, textvariable=var, width=8,
                              justify="center")
            entry.grid(row=row, column=col_base + 1, sticky="w", **gpad)

            # 绑定回车和失焦自动应用
            entry.bind("<Return>", lambda e: self._apply_params())
            entry.bind("<FocusOut>", lambda e: self._apply_params())

            # 提示 (鼠标悬停)
            if tip:
                self._create_tooltip(entry, tip)

        # 按钮行
        total_rows = (len(tunable_params) + cols_per_row - 1) // cols_per_row
        btn_frame = ttk.Frame(self.frm_params)
        btn_frame.grid(row=total_rows, column=0, columnspan=6,
                       pady=(5, 5), sticky="e", padx=10)

        self.btn_apply = ttk.Button(btn_frame, text=i18n._("btn_apply"),
                   command=self._apply_params, width=10)
        self.btn_apply.pack(side="left", padx=3)
        self.btn_reset = ttk.Button(btn_frame, text=i18n._("btn_reset"),
                   command=self._reset_params, width=10)
        self.btn_reset.pack(side="left", padx=3)

    def _config_to_display(self, attr: str, vtype: str) -> str:
        """将 config 值转换为 GUI 显示值"""
        val = getattr(config, attr)
        if vtype == "ms":
            return str(round(val * 1000))       # 秒 → 毫秒
        elif vtype == "pct":
            return str(round(val * 100))        # 0.55 → 55
        elif vtype == "int":
            return str(int(val))
        elif vtype == "float":
            # 自动选择合理的小数位
            if val == 0:
                return "0"
            elif abs(val) < 0.001:
                return f"{val:.5f}"
            elif abs(val) < 0.1:
                return f"{val:.4f}"
            elif abs(val) < 10:
                return f"{val:.3f}"
            else:
                return f"{val:.1f}"
        return str(val)

    def _display_to_config(self, text: str, vtype: str):
        """将 GUI 显示值转换为 config 值"""
        text = text.strip()
        if not text:
            return None
        try:
            if vtype == "ms":
                return float(text) / 1000.0     # 毫秒 → 秒
            elif vtype == "pct":
                return float(text) / 100.0      # 55 → 0.55
            elif vtype == "int":
                return int(float(text))
            elif vtype == "float":
                return float(text)
        except ValueError:
            return None
        return None

    def _apply_params(self, save_only=False):
        """读取所有参数输入框，应用到 config 并保存到文件
        
        Args:
            save_only: 如果为True，只保存到config但不写入文件和输出日志
        """
        changed = []
        for attr, (var, vtype) in self._param_vars.items():
            new_val = self._display_to_config(var.get(), vtype)
            if new_val is None:
                continue

            old_val = getattr(config, attr)
            if vtype == "ms":
                is_same = abs(old_val - new_val) < 0.0001
            elif vtype == "float":
                is_same = abs(old_val - new_val) < 1e-7
            else:
                is_same = old_val == new_val

            if not is_same:
                setattr(config, attr, new_val)
                changed.append(f"{attr}: {old_val} → {new_val}")

        if changed and not save_only:
            self._save_settings()
            self._log_msg(f"[参数] 已更新并保存: {', '.join(changed)}")

    def _reset_params(self):
        """恢复所有参数到默认值并删除配置文件"""
        defaults = {
            "BITE_FORCE_HOOK":  0.500,
            "FISH_GAME_SIZE":   30,
            "DEAD_ZONE":        12,
            "HOLD_MIN_S":       0.015,
            "HOLD_MAX_S":       0.100,
            "HOLD_GAIN":        0.040,
            "PREDICT_AHEAD":    0.400,
            "SPEED_DAMPING":    0.00025,
            "MAX_FISH_BAR_DIST": 300,
            "VELOCITY_SMOOTH":  0.5,
            "TRACK_MIN_ANGLE":  3.0,
            "TRACK_MAX_ANGLE":  45.0,
            "REGION_UP":        300,
            "REGION_DOWN":      400,
            "REGION_X":         100,
            "POST_CATCH_DELAY": 2.800,
            "SHAKE_HEAD_TIME":  0.0300,
            "INITIAL_PRESS_TIME": 0.2,
            "VERIFY_CONSECUTIVE": 1,
            "SUCCESS_PROGRESS": 0.42,
        }

        for attr, default_val in defaults.items():
            setattr(config, attr, default_val)
            if attr in self._param_vars:
                var, vtype = self._param_vars[attr]
                var.set(self._config_to_display(attr, vtype))

        # 删除配置文件
        try:
            import os
            if os.path.exists(config.SETTINGS_FILE):
                os.remove(config.SETTINGS_FILE)
        except Exception:
            pass
        self._log_msg("[参数] 已恢复默认值")

    # ══════════════════════════════════════════════════════
    #  参数持久化
    # ══════════════════════════════════════════════════════

    def _save_settings(self):
        """将当前可调参数保存到 settings.json"""
        data = {}
        for attr, (_, vtype) in self._param_vars.items():
            data[attr] = getattr(config, attr)
        data["DETECT_ROI"] = config.DETECT_ROI
        data["YOLO_COLLECT"] = config.YOLO_COLLECT
        data["YOLO_COLLECT_ON_FAIL"] = config.YOLO_COLLECT_ON_FAIL
        data["YOLO_DEVICE"] = config.YOLO_DEVICE
        data["SHOW_DEBUG"] = config.SHOW_DEBUG
        data["FISH_WHITELIST"] = config.FISH_WHITELIST
        data["ENABLE_FORCE_RESET"] = config.ENABLE_FORCE_RESET
        data["LANGUAGE"] = getattr(config, 'LANGUAGE', 'zh')
        try:
            with open(config.SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self._log_msg(f"[警告] 保存配置失败: {e}")

    def _load_settings(self):
        """启动时从 settings.json 加载参数"""
        import os
        if not os.path.exists(config.SETTINGS_FILE):
            return
        try:
            with open(config.SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("HOLD_GAIN", 1) < 0.02:
                data["HOLD_GAIN"] = 0.040
            if data.get("SPEED_DAMPING", 0) > 0.001:
                data["SPEED_DAMPING"] = 0.00025
            if data.get("HOLD_MAX_S", 1) < 0.08:
                data["HOLD_MAX_S"] = 0.100
            if data.get("HOLD_MIN_S", 1) < 0.01:
                data["HOLD_MIN_S"] = 0.015

            loaded = []
            for attr, val in data.items():
                if attr == "DETECT_ROI":
                    if val and isinstance(val, list) and len(val) == 4:
                        config.DETECT_ROI = val
                        if hasattr(self, 'var_roi'):
                            x, y, w, h = val
                            self.var_roi.set(f"X={x} Y={y} {w}x{h}")
                            self.lbl_roi.config(foreground="green")
                    else:
                        config.DETECT_ROI = None
                    loaded.append(attr)
                elif attr == "YOLO_COLLECT":
                    config.YOLO_COLLECT = bool(val)
                    if hasattr(self, 'var_yolo_collect'):
                        self.var_yolo_collect.set(config.YOLO_COLLECT)
                    loaded.append(attr)
                elif attr == "YOLO_COLLECT_ON_FAIL":
                    config.YOLO_COLLECT_ON_FAIL = bool(val)
                    if hasattr(self, 'var_yolo_collect_on_fail'):
                        self.var_yolo_collect_on_fail.set(config.YOLO_COLLECT_ON_FAIL)
                    loaded.append(attr)
                elif attr == "YOLO_DEVICE":
                    if val in ("auto", "cpu", "gpu"):
                        config.YOLO_DEVICE = val
                        if hasattr(self, 'var_yolo_device'):
                            self.var_yolo_device.set(val)
                    loaded.append(attr)
                elif attr == "SHOW_DEBUG":
                    config.SHOW_DEBUG = bool(val)
                    if hasattr(self, 'var_show_debug'):
                        self.var_show_debug.set(config.SHOW_DEBUG)
                    loaded.append(attr)
                elif attr == "FISH_WHITELIST":
                    if isinstance(val, dict):
                        config.FISH_WHITELIST.update(val)
                    loaded.append(attr)
                elif attr == "ENABLE_FORCE_RESET":
                    config.ENABLE_FORCE_RESET = bool(val)
                    if hasattr(self, 'var_force_reset'):
                        self.var_force_reset.set(config.ENABLE_FORCE_RESET)
                    loaded.append(attr)
                elif attr == "LANGUAGE":
                    if val in ("zh", "en", "jp"):
                        config.LANGUAGE = val
                        self._current_lang = val
                        i18n.set_language(val)
                        if hasattr(self, 'var_language'):
                            self.var_language.set(val)
                    loaded.append(attr)
                elif attr in self._param_vars:
                    setattr(config, attr, val)
                    var, vtype = self._param_vars[attr]
                    var.set(self._config_to_display(attr, vtype))
                    loaded.append(attr)
            if loaded:
                pass
        except Exception as e:
            self._log_msg(f"[警告] 加载配置失败: {e}")

    @staticmethod
    def _create_tooltip(widget, text: str):
        """为控件创建鼠标悬停提示"""
        tip_window = [None]

        def show(event):
            if tip_window[0]:
                return
            tw = tk.Toplevel(widget)
            tw.wm_overrideredirect(True)
            tw.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            lbl = tk.Label(tw, text=text, background="#ffffe0",
                           relief="solid", borderwidth=1,
                           font=("", 9), padx=4, pady=2)
            lbl.pack()
            tip_window[0] = tw

        def hide(_):
            if tip_window[0]:
                tip_window[0].destroy()
                tip_window[0] = None

        widget.bind("<Enter>", show)
        widget.bind("<Leave>", hide)

    # ══════════════════════════════════════════════════════
    #  按钮回调
    # ══════════════════════════════════════════════════════

    @staticmethod
    def _has_non_ascii(path: str) -> bool:
        try:
            path.encode("ascii")
            return False
        except UnicodeEncodeError:
            return True

    def _on_start(self):
        """开始钓鱼"""
        if self.bot.running:
            return

        if self._has_non_ascii(config.BASE_DIR):
            self._log_msg("[错误] 程序所在路径包含中文或特殊字符，会导致图片/模型加载失败！")
            self._log_msg(f"  当前路径: {config.BASE_DIR}")
            self._log_msg("  请将程序移动到纯英文路径下再运行，例如: D:\\fish")
            return

        # 先尝试连接窗口
        if not self.bot.window.is_valid():
            if not self.bot.window.find():
                self._log_msg("[错误] 未找到 VRChat 窗口！请确保游戏正在运行。")
                return

        self.var_window.set(f"{self.bot.window.title} (HWND={self.bot.window.hwnd})")

        # ★ 开始前应用一次参数 (确保 GUI 上的值生效)
        self._apply_params()

        self.bot.running = True
        self.bot.state = i18n._("running")

        # 启动后台线程
        if self.bot_thread is None or not self.bot_thread.is_alive():
            self.bot_thread = threading.Thread(target=self.bot.run, daemon=True)
            self.bot_thread.start()

        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self._log_msg("[系统] ▶ 开始自动钓鱼")

    def _on_stop(self):
        """停止钓鱼"""
        self.bot.running = False
        self.bot.state = "stopped"  # 使用统一的状态键
        self.bot.input.safe_release()
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self._log_msg("[系统] ■ 已停止")
        self._save_log()

    def _on_toggle_debug(self):
        """切换调试模式"""
        self.bot.debug_mode = not self.bot.debug_mode
        # 使用统一的开启/关闭翻译
        tag = i18n._("debug_on") if self.bot.debug_mode else i18n._("debug_off")
        self.var_debug.set(tag)
        self._log_msg(f"[系统] 调试模式: {tag}")
        if self.bot.debug_mode:
            self._log_msg("[提示] 调试截图将保存到 debug/ 目录，检测器将输出置信度")

    def _on_connect(self):
        """手动连接 VRChat 窗口"""
        if self.bot.window.find():
            self.var_window.set(
                f"{self.bot.window.title} (HWND={self.bot.window.hwnd})"
            )
            # 重置截图方式，下次截图时重新检测 PrintWindow
            self.bot.screen.reset_capture_method()
            self._log_msg(f"[系统] 已连接: {self.bot.window.title}")
        else:
            self.var_window.set(i18n._("not_found"))
            self._log_msg("[错误] 未找到 VRChat 窗口")

    def _on_screenshot(self):
        """手动保存当前截图（调试用）"""
        if not self.bot.window.is_valid():
            if not self.bot.window.find():
                self._log_msg("[错误] 无法截图: 未连接 VRChat 窗口")
                return

        img, region = self.screen_capture_safe()
        if img is not None:
            self.bot.screen.save_debug(img, "manual_screenshot")
            h, w = img.shape[:2]
            self._log_msg(f"[截图] 已保存 ({w}×{h}) → debug/manual_screenshot.png")
            if region:
                self._log_msg(f"       窗口区域: x={region[0]} y={region[1]} w={region[2]} h={region[3]}")
        else:
            self._log_msg("[错误] 截图失败")

    def _on_clear_log(self):
        """清空日志文本框"""
        self.txt_log.config(state="normal")
        self.txt_log.delete("1.0", "end")
        self.txt_log.config(state="disabled")

    def _on_whitelist(self):
        """弹窗: 勾选要钓的鱼种"""
        FISH_NAMES = [
            ("fish_black",   i18n._("fish_black")),
            ("fish_white",   i18n._("fish_white")),
            ("fish_copper",  i18n._("fish_copper")),
            ("fish_green",   i18n._("fish_green")),
            ("fish_blue",    i18n._("fish_blue")),
            ("fish_purple",  i18n._("fish_purple")),
            ("fish_pink",    i18n._("fish_pink")),
            ("fish_red",     i18n._("fish_red")),
            ("fish_rainbow", i18n._("fish_rainbow")),
        ]
        win = tk.Toplevel(self.root)
        win.title(i18n._("whitelist_title"))
        win.geometry("200x320")
        win.resizable(False, False)
        win.transient(self.root)
        win.grab_set()

        ttk.Label(win, text=i18n._("whitelist_label")).pack(pady=(10, 5))

        wl = config.FISH_WHITELIST
        chk_vars = {}
        for key, name in FISH_NAMES:
            var = tk.BooleanVar(value=wl.get(key, True))
            chk_vars[key] = var
            ttk.Checkbutton(win, text=name, variable=var).pack(
                anchor="w", padx=30)

        def _apply():
            for key, var in chk_vars.items():
                config.FISH_WHITELIST[key] = var.get()
            self._save_settings()
            enabled = [n for (k, n) in FISH_NAMES if chk_vars[k].get()]
            self._log_msg(f"[白名单] 已更新: {', '.join(enabled)}")
            win.destroy()

        ttk.Button(win, text=i18n._("btn_confirm"), command=_apply).pack(pady=10)

    def _on_topmost(self):
        """切换窗口置顶 (用 int 0/1 确保兼容性)"""
        topmost = self.var_topmost.get()
        self.root.wm_attributes("-topmost", 1 if topmost else 0)
        if not topmost:
            self.root.lift()
            self.root.focus_force()

    def _on_debug_toggle(self):
        """切换 debug 窗口显示"""
        config.SHOW_DEBUG = self.var_show_debug.get()
        self._save_settings()
        state = "开启" if config.SHOW_DEBUG else "关闭 (提升性能)"
        self._log_msg(f"[Debug] 调试窗口: {state}")
        if not config.SHOW_DEBUG:
            try:
                import cv2
                cv2.destroyWindow("Debug Overlay")
            except Exception:
                pass

    def _on_force_reset_toggle(self):
        """切换强制重置功能"""
        config.ENABLE_FORCE_RESET = self.var_force_reset.get()
        self._save_settings()
        state = "启用" if config.ENABLE_FORCE_RESET else "禁用"
        self._log_msg(f"[设置] 强制重置功能: {state} "
                      f"(连续{config.MAX_RETRY_NO_MINIGAME}次未检测到小游戏等待{config.FORCE_RESET_DELAY}s)")

    def _on_view_reset_log(self):
        """查看强制重置日志弹窗"""
        log_window = tk.Toplevel(self.root)
        log_window.title(i18n._("reset_log_title"))
        log_window.geometry("450x500")
        log_window.transient(self.root)  # 设置为父窗口的子窗口
        log_window.grab_set()  # 模态窗口
        
        # 标题
        ttk.Label(log_window, text=i18n._("reset_log_record"), font=("", 12, "bold")).pack(pady=10)
        
        # 获取日志
        reset_log = self.bot.get_force_reset_log()
        
        if not reset_log:
            ttk.Label(log_window, text=i18n._("reset_log_empty"), foreground="gray").pack(pady=20)
        else:
            # 创建主框架
            main_frame = ttk.Frame(log_window)
            main_frame.pack(fill="both", expand=True, padx=10, pady=5)
            
            # 表头
            frm_header = ttk.Frame(main_frame)
            frm_header.pack(fill="x", pady=(0, 5))
            ttk.Label(frm_header, text=i18n._("reset_log_index"), width=10).pack(side="left")
            ttk.Label(frm_header, text=i18n._("reset_log_time"), width=20).pack(side="left")
            
            # 分隔线
            ttk.Separator(main_frame, orient="horizontal").pack(fill="x")
            
            # 创建滚动区域 - 使用ScrolledText简化实现
            txt_frame = ttk.Frame(main_frame)
            txt_frame.pack(fill="both", expand=True, pady=5)
            
            txt_records = tk.Text(txt_frame, height=15, width=40, 
                                  font=("Consolas", 10),
                                  bg="#f5f5f5", fg="#333333",
                                  relief="flat", state="disabled")
            scrollbar = ttk.Scrollbar(txt_frame, orient="vertical", command=txt_records.yview)
            txt_records.configure(yscrollcommand=scrollbar.set)
            
            txt_records.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # 填充数据（倒序显示，最新的在前）
            txt_records.config(state="normal")
            for entry in reversed(reset_log):
                line = f"#{entry['count']:<6} {entry['timestamp']}\n"
                txt_records.insert("end", line)
            txt_records.config(state="disabled")
        
        # 统计信息
        ttk.Separator(log_window, orient="horizontal").pack(fill="x", padx=10, pady=5)
        frm_stats = ttk.Frame(log_window)
        frm_stats.pack(fill="x", padx=10, pady=5)
        ttk.Label(frm_stats, text=i18n._("reset_log_total", len(reset_log)), 
                  font=("", 10, "bold")).pack(side="left")
        
        # 按钮区域
        frm_buttons = ttk.Frame(log_window)
        frm_buttons.pack(fill="x", padx=10, pady=10)
        
        btn_clear = ttk.Button(frm_buttons, text=i18n._("btn_clear"),
                   command=lambda: self._on_clear_reset_log(log_window))
        btn_clear.pack(side="left", padx=5)
        btn_close = ttk.Button(frm_buttons, text=i18n._("btn_close"),
                   command=log_window.destroy)
        btn_close.pack(side="right", padx=5)

    def _on_clear_reset_log(self, parent_window=None):
        """清空强制重置日志"""
        import tkinter.messagebox as msgbox
        if msgbox.askyesno(i18n._("confirm_title"), i18n._("confirm_clear_log")):
            self.bot.clear_force_reset_log()
            self.var_force_reset_count.set("0")
            self._log_msg("[日志] 强制重置日志已清空")
            if parent_window:
                parent_window.destroy()

    def _preload_yolo(self):
        """后台线程预加载 YOLO 模型，避免阻塞 GUI"""
        def _load():
            try:
                from core.bot import _get_yolo_detector
                self.bot.yolo = _get_yolo_detector()
                pass
            except Exception as e:
                self._log_msg(f"[YOLO] 预加载失败: {e}")
        t = threading.Thread(target=_load, daemon=True)
        t.start()

    def _on_yolo_collect_toggle(self):
        """切换 YOLO 数据采集模式"""
        collect = self.var_yolo_collect.get()
        config.YOLO_COLLECT = collect
        self._save_settings()
        if collect:
            self._log_msg(
                "[YOLO] 数据采集已开启 — 钓鱼时将自动保存截图"
            )
        else:
            self._log_msg("[YOLO] 数据采集已关闭")

    def _on_yolo_collect_on_fail_toggle(self):
        """切换仅在失败时采集数据模式"""
        on_fail = self.var_yolo_collect_on_fail.get()
        config.YOLO_COLLECT_ON_FAIL = on_fail
        self._save_settings()
        if on_fail:
            self._log_msg("[YOLO] 仅在钓鱼失败时采集图像")
        else:
            self._log_msg("[YOLO] 恢复所有钓鱼过程采集图像")

    def _on_yolo_device_change(self, _event=None):
        """切换 YOLO 推理设备"""
        dev = self.var_yolo_device.get()
        config.YOLO_DEVICE = dev
        self._save_settings()
        labels = {"auto": "自动 (优先GPU)", "cpu": "CPU (不占显卡)",
                  "gpu": "GPU (需要CUDA)"}
        self._log_msg(f"[YOLO] 设备已切换: {labels.get(dev, dev)} — 下次启动生效")

    def _update_yolo_status(self):
        """更新 YOLO 状态显示"""
        import os as _os
        model_ok = _os.path.exists(config.YOLO_MODEL)
        unlabeled = _os.path.join(
            config.BASE_DIR, "yolo", "dataset", "images", "unlabeled")
        train = _os.path.join(
            config.BASE_DIR, "yolo", "dataset", "images", "train")
        n_unlabeled = len([
            f for f in _os.listdir(unlabeled)
            if f.endswith((".png", ".jpg"))
        ]) if _os.path.isdir(unlabeled) else 0
        n_train = len([
            f for f in _os.listdir(train)
            if f.endswith((".png", ".jpg"))
        ]) if _os.path.isdir(train) else 0

        parts = []
        if model_ok:
            parts.append(i18n._("yolo_model_ok"))
        else:
            parts.append(i18n._("yolo_model_fail"))
        parts.append(f"train:{n_train}")
        parts.append(f"unlabeled:{n_unlabeled}")
        self.var_yolo_status.set(" | ".join(parts))

    def _on_select_roi(self):
        """框选钓鱼UI检测区域"""
        if not self.bot.window.is_valid():
            if not self.bot.window.find():
                self._log_msg("[错误] 请先连接 VRChat 窗口")
                return

        img, _ = self.screen_capture_safe()
        if img is None:
            self._log_msg("[错误] 截图失败, 无法框选")
            return

        self._log_msg(
            "[框选] 请在弹出窗口中用鼠标框选钓鱼UI区域, "
            "按回车确认, 按ESC取消"
        )

        win_name = "Select Fishing ROI - Enter=OK / Esc=Cancel"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        h, w = img.shape[:2]
        dw = min(w, 1280)
        dh = int(h * dw / w)
        cv2.resizeWindow(win_name, dw, dh)

        roi = cv2.selectROI(win_name, img,
                            fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()

        x, y, w_r, h_r = [int(v) for v in roi]
        if w_r > 10 and h_r > 10:
            config.DETECT_ROI = [x, y, w_r, h_r]
            self._save_settings()
            self.var_roi.set(f"X={x} Y={y} {w_r}x{h_r}")
            self.lbl_roi.config(foreground="green")
            self._log_msg(
                f"[框选] ✓ 检测区域已设置: X={x} Y={y} {w_r}x{h_r}"
            )
        else:
            self._log_msg("[框选] 已取消 (区域太小或按了ESC)")

    def _on_clear_roi(self):
        """清除框选区域"""
        config.DETECT_ROI = None
        self._save_settings()
        self.var_roi.set(i18n._("roi_not_set"))
        self.lbl_roi.config(foreground="gray")
        self._log_msg("[框选] 已清除检测区域, 将使用全屏搜索")

    def screen_capture_safe(self):
        """安全截取屏幕"""
        try:
            return self.bot.screen.grab_window(self.bot.window)
        except Exception as e:
            self._log_msg(f"[错误] 截图异常: {e}")
            return None, None

    # ══════════════════════════════════════════════════════
    #  快捷键回调（从 VRChat 中触发）
    # ══════════════════════════════════════════════════════

    def _toggle_from_hotkey(self):
        """F9 — 切换开始/停止"""
        if self.bot.running:
            self.root.after(0, self._on_stop)
        else:
            self.root.after(0, self._on_start)

    def _stop_from_hotkey(self):
        """F10 — 停止"""
        self.root.after(0, self._on_stop)

    def _toggle_debug_from_hotkey(self):
        """F11 — 调试"""
        self.root.after(0, self._on_toggle_debug)

    # ══════════════════════════════════════════════════════
    #  轮询更新
    # ══════════════════════════════════════════════════════

    def _poll(self):
        """每 100ms 从日志队列读取消息，更新状态面板"""
        # 读取日志
        try:
            for _ in range(20):          # 每次最多处理 20 条
                msg = log.log_queue.get_nowait()
                self._append_log(msg)
        except queue.Empty:
            pass

        # 更新状态（使用当前语言翻译）
        if self.bot.running:
            self.var_state.set(i18n._("running"))
        else:
            # 将bot的内部状态映射到翻译键
            state_map = {
                # 就绪状态
                "就绪": "ready",
                "ready": "ready",
                "Ready": "ready", 
                "準備完了": "ready",
                # 运行中状态
                "运行中": "running",
                "running": "running",
                "Running": "running",
                "実行中": "running",
                # 已停止状态
                "已停止": "stopped",
                "stopped": "stopped",
                "Stopped": "stopped",
                "停止": "stopped",
            }
            # 获取状态对应的翻译键，如果没有则使用ready
            state_key = state_map.get(self.bot.state, "ready")
            self.var_state.set(i18n._(state_key))
        self.var_count.set(str(self.bot.fish_count))

        # 更新成功率显示
        total = self.bot.success_count + self.bot.fail_count
        if total > 0:
            rate = self.bot.success_count / total * 100
            self.var_success_rate.set(f"{self.bot.success_count}/{total} ({rate:.1f}%)")
        else:
            self.var_success_rate.set("0/0 (0%)")

        # 更新强制重置次数
        self.var_force_reset_count.set(str(self.bot._force_reset_count))

        # 状态颜色
        if self.bot.running:
            self.lbl_state.config(foreground="green")
        else:
            self.lbl_state.config(foreground="gray")

        # 检测线程是否意外退出
        if self.bot_thread and not self.bot_thread.is_alive() and self.bot.running:
            self.bot.running = False
            self.btn_start.config(state="normal")
            self.btn_stop.config(state="disabled")

        self.root.after(100, self._poll)

    # ══════════════════════════════════════════════════════
    #  日志输出
    # ══════════════════════════════════════════════════════

    def _log_msg(self, msg: str):
        """直接向日志区写入（不经过 logger queue）"""
        import time
        ts = time.strftime("%H:%M:%S")
        self._append_log(f"[{ts}] {msg}")

    def _append_log(self, text: str):
        """向日志文本框追加一行"""
        self.txt_log.config(state="normal")
        self.txt_log.insert("end", text + "\n")
        self.txt_log.see("end")
        self.txt_log.config(state="disabled")

    # ══════════════════════════════════════════════════════
    #  关闭
    # ══════════════════════════════════════════════════════

    def _on_close(self):
        """窗口关闭处理"""
        self.bot.running = False
        self.bot.input.safe_release()
        self._save_settings()
        self._save_log()
        self.root.destroy()

    def _save_log(self):
        """保存日志到文件 (覆盖上一次)"""
        import os
        path = os.path.join(config.DEBUG_DIR, "last_run.log")
        log.save(path)
        self._log_msg(i18n._("msg_log_saved", path))
