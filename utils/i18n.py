"""
国际化 (i18n) 模块
==================
支持多语言切换：中文、英语、日语
"""

import json
import os

# 默认语言
DEFAULT_LANG = "zh"

# 翻译字典
TRANSLATIONS = {
    "zh": {
        # 窗口标题
        "app_title": "VRC auto fish 263302",
        "reset_log_title": "强制重置日志",
        "whitelist_title": "钓鱼白名单",
        
        # 状态面板
        "status_group": " 状态 ",
        "running_status": "运行状态:",
        "window_status": "VRChat窗口:",
        "fish_count": "钓鱼次数:",
        "success_rate": "成功率:",
        "force_reset": "强制重置:",
        "debug_mode": "调试模式:",
        "ready": "就绪",
        "running": "运行中",
        "stopped": "已停止",
        "connected": "已连接",
        "not_connected": "未连接",
        "not_found": "未找到",
        
        # 按钮
        "btn_start": "▶ 开始 (F9)",
        "btn_stop": "■ 停止 (F10)",
        "btn_debug": "调试模式 (F11)",
        "btn_connect": "🔗 连接窗口",
        "btn_screenshot": "📸 保存截图",
        "btn_clear_log": "🗑 清空日志",
        "btn_whitelist": "🐟 白名单",
        "btn_view_log": "查看日志",
        "btn_clear": "清空日志",
        "btn_close": "关闭",
        "btn_confirm": "确定",
        "btn_apply": "应用参数",
        "btn_reset": "恢复默认",
        
        # 选项
        "opt_topmost": "窗口置顶",
        "opt_debug_window": "Debug窗口",
        "opt_force_reset": "强制重置",
        "opt_collect_data": "采集数据",
        "opt_collect_on_fail": "仅在失败采集",
        
        # YOLO
        "yolo_group": " YOLO 目标检测 ",
        "yolo_enabled": "YOLO 已启用",
        "yolo_device": "设备:",
        "yolo_model_ok": "模型 ✓",
        "yolo_model_fail": "模型 ✗",
        
        # 区域选择
        "btn_select_roi": "📐 框选检测区域",
        "btn_clear_roi": "✕ 清除区域",
        "detect_region": "检测区域:",
        "roi_not_set": "未设置 (全屏搜索)",
        
        # 参数面板
        "params_group": " 小游戏参数 (实时生效) ",
        
        # 日志
        "log_group": " 日志 ",
        "debug_on": "开启",
        "debug_off": "关闭",
        "reset_log_record": "强制重置记录",
        "reset_log_empty": "暂无强制重置记录",
        "reset_log_index": "序号",
        "reset_log_time": "重置时间",
        "reset_log_total": "总计: {} 次",
        
        # 白名单
        "whitelist_label": "勾选要钓的鱼:",
        "fish_black": "黑鱼",
        "fish_white": "白鱼",
        "fish_copper": "铜鱼",
        "fish_green": "绿鱼",
        "fish_blue": "蓝鱼",
        "fish_purple": "紫鱼",
        "fish_pink": "粉鱼",
        "fish_red": "红鱼",
        "fish_rainbow": "彩鱼",
        
        # 语言选择
        "language": "语言:",
        "lang_zh": "中文",
        "lang_en": "English",
        "lang_jp": "日本語",
        
        # 消息
        "msg_github": "GitHub: https://github.com/day123123123/vrc-auto-fish",
        "msg_log_saved": "[系统] 日志已保存 → {}",
        "msg_settings_saved": "[系统] 配置已保存",
        "msg_settings_loaded": "[系统] 配置已加载",
        
        # 确认对话框
        "confirm_clear_log": "确定要清空所有强制重置日志吗？",
        "confirm_title": "确认",
    },
    "en": {
        # Window titles
        "app_title": "VRC auto fish 263302",
        "reset_log_title": "Force Reset Log",
        "whitelist_title": "Fish Whitelist",
        
        # Status panel
        "status_group": " Status ",
        "running_status": "Status:",
        "window_status": "VRChat Window:",
        "fish_count": "Fish Count:",
        "success_rate": "Success Rate:",
        "force_reset": "Force Reset:",
        "debug_mode": "Debug Mode:",
        "ready": "Ready",
        "running": "Running",
        "stopped": "Stopped",
        "connected": "Connected",
        "not_connected": "Not Connected",
        "not_found": "Not Found",
        
        # Buttons
        "btn_start": "▶ Start (F9)",
        "btn_stop": "■ Stop (F10)",
        "btn_debug": "Debug Mode (F11)",
        "btn_connect": "🔗 Connect Window",
        "btn_screenshot": "📸 Screenshot",
        "btn_clear_log": "🗑 Clear Log",
        "btn_whitelist": "🐟 Whitelist",
        "btn_view_log": "View Log",
        "btn_clear": "Clear Log",
        "btn_close": "Close",
        "btn_confirm": "OK",
        "btn_apply": "Apply",
        "btn_reset": "Reset",
        
        # Options
        "opt_topmost": "Always on Top",
        "opt_debug_window": "Debug Window",
        "opt_force_reset": "Force Reset",
        "opt_collect_data": "Collect Data",
        "opt_collect_on_fail": "Collect on Fail Only",
        
        # YOLO
        "yolo_group": " YOLO Detection ",
        "yolo_enabled": "YOLO Enabled",
        "yolo_device": "Device:",
        "yolo_model_ok": "Model ✓",
        "yolo_model_fail": "Model ✗",
        
        # ROI Selection
        "btn_select_roi": "📐 Select ROI",
        "btn_clear_roi": "✕ Clear ROI",
        "detect_region": "Detect Region:",
        "roi_not_set": "Not Set (Full Screen)",
        
        # Params panel
        "params_group": " Minigame Params (Live) ",
        
        # Log
        "log_group": " Log ",
        "debug_on": "On",
        "debug_off": "Off",
        "reset_log_record": "Force Reset Records",
        "reset_log_empty": "No force reset records",
        "reset_log_index": "No.",
        "reset_log_time": "Time",
        "reset_log_total": "Total: {} times",
        
        # Whitelist
        "whitelist_label": "Select fish to catch:",
        "fish_black": "Black Fish",
        "fish_white": "White Fish",
        "fish_copper": "Copper Fish",
        "fish_green": "Green Fish",
        "fish_blue": "Blue Fish",
        "fish_purple": "Purple Fish",
        "fish_pink": "Pink Fish",
        "fish_red": "Red Fish",
        "fish_rainbow": "Rainbow Fish",
        
        # Language
        "language": "Language:",
        "lang_zh": "中文",
        "lang_en": "English",
        "lang_jp": "日本語",
        
        # Messages
        "msg_github": "GitHub: https://github.com/day123123123/vrc-auto-fish",
        "msg_log_saved": "[System] Log saved → {}",
        "msg_settings_saved": "[System] Settings saved",
        "msg_settings_loaded": "[System] Settings loaded",
        
        # Confirm dialogs
        "confirm_clear_log": "Are you sure you want to clear all force reset logs?",
        "confirm_title": "Confirm",
    },
    "jp": {
        # ウィンドウタイトル
        "app_title": "VRC auto fish 263302",
        "reset_log_title": "強制リセットログ",
        "whitelist_title": "魚ホワイトリスト",
        
        # ステータスパネル
        "status_group": " ステータス ",
        "running_status": "実行状態:",
        "window_status": "VRChatウィンドウ:",
        "fish_count": "釣り回数:",
        "success_rate": "成功率:",
        "force_reset": "強制リセット:",
        "debug_mode": "デバッグモード:",
        "ready": "準備完了",
        "running": "実行中",
        "stopped": "停止",
        "connected": "接続済み",
        "not_connected": "未接続",
        "not_found": "未検出",
        
        # ボタン
        "btn_start": "▶ 開始 (F9)",
        "btn_stop": "■ 停止 (F10)",
        "btn_debug": "デバッグモード (F11)",
        "btn_connect": "🔗 ウィンドウ接続",
        "btn_screenshot": "📸 スクリーンショット",
        "btn_clear_log": "🗑 ログ消去",
        "btn_whitelist": "🐟 ホワイトリスト",
        "btn_view_log": "ログ表示",
        "btn_clear": "ログ消去",
        "btn_close": "閉じる",
        "btn_confirm": "OK",
        "btn_apply": "適用",
        "btn_reset": "リセット",
        
        # オプション
        "opt_topmost": "常に手前に表示",
        "opt_debug_window": "デバッグウィンドウ",
        "opt_force_reset": "強制リセット",
        "opt_collect_data": "データ収集",
        "opt_collect_on_fail": "失敗時のみ収集",
        
        # YOLO
        "yolo_group": " YOLO 物体検出 ",
        "yolo_enabled": "YOLO 有効",
        "yolo_device": "デバイス:",
        "yolo_model_ok": "モデル ✓",
        "yolo_model_fail": "モデル ✗",
        
        # 領域選択
        "btn_select_roi": "📐 検出領域選択",
        "btn_clear_roi": "✕ 領域消去",
        "detect_region": "検出領域:",
        "roi_not_set": "未設定 (全画面)",
        
        # パラメータパネル
        "params_group": " ミニゲーム設定 (リアルタイム) ",
        
        # ログ
        "log_group": " ログ ",
        "debug_on": "オン",
        "debug_off": "オフ",
        "reset_log_record": "強制リセット記録",
        "reset_log_empty": "強制リセット記録なし",
        "reset_log_index": "番号",
        "reset_log_time": "時刻",
        "reset_log_total": "合計: {} 回",
        
        # ホワイトリスト
        "whitelist_label": "釣る魚を選択:",
        "fish_black": "黒魚",
        "fish_white": "白魚",
        "fish_copper": "銅魚",
        "fish_green": "緑魚",
        "fish_blue": "青魚",
        "fish_purple": "紫魚",
        "fish_pink": "桃魚",
        "fish_red": "赤魚",
        "fish_rainbow": "虹魚",
        
        # 言語
        "language": "言語:",
        "lang_zh": "中文",
        "lang_en": "English",
        "lang_jp": "日本語",
        
        # メッセージ
        "msg_github": "GitHub: https://github.com/day123123123/vrc-auto-fish",
        "msg_log_saved": "[システム] ログ保存 → {}",
        "msg_settings_saved": "[システム] 設定保存",
        "msg_settings_loaded": "[システム] 設定読込",
        
        # 確認ダイアログ
        "confirm_clear_log": "強制リセットログを消去しますか？",
        "confirm_title": "確認",
    }
}


class I18n:
    """国际化支持类"""
    
    _instance = None
    _current_lang = DEFAULT_LANG
    _listeners = []
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def set_language(self, lang: str):
        """设置当前语言"""
        if lang in TRANSLATIONS:
            self._current_lang = lang
            # 通知所有监听器
            for listener in self._listeners:
                try:
                    listener()
                except Exception:
                    pass
    
    def get_language(self) -> str:
        """获取当前语言"""
        return self._current_lang
    
    def get_text(self, key: str, *args) -> str:
        """获取翻译文本"""
        lang_dict = TRANSLATIONS.get(self._current_lang, TRANSLATIONS[DEFAULT_LANG])
        text = lang_dict.get(key, key)
        if args:
            try:
                text = text.format(*args)
            except Exception:
                pass
        return text
    
    def add_listener(self, callback):
        """添加语言变化监听器"""
        if callback not in self._listeners:
            self._listeners.append(callback)
    
    def remove_listener(self, callback):
        """移除语言变化监听器"""
        if callback in self._listeners:
            self._listeners.remove(callback)


# 全局实例
_i18n = I18n()

# 便捷函数
def set_language(lang: str):
    """设置当前语言"""
    _i18n.set_language(lang)

def get_language() -> str:
    """获取当前语言"""
    return _i18n.get_language()

def get_text(key: str, *args) -> str:
    """获取翻译文本"""
    return _i18n.get_text(key, *args)

# 简写方式 _(key) 获取翻译
_ = get_text

def add_listener(callback):
    """添加语言变化监听器"""
    _i18n.add_listener(callback)

def remove_listener(callback):
    """移除语言变化监听器"""
    _i18n.remove_listener(callback)


# 可调参数的翻译（按语言）
TUNABLE_PARAMS_I18N = {
    "zh": [
        ("强制提竿(s)",   "BITE_FORCE_HOOK",  "float", "从抛竿到鱼儿咬钩的等待时间"),
        ("鱼像素大小",    "FISH_GAME_SIZE",   "int",   "游戏内鱼图标的大致像素,越小搜索倍率越高"),
        ("死区(px)",      "DEAD_ZONE",        "int",   "越大越容易触发按住"),
        ("抗重力基准(ms)","HOLD_MIN_S",       "ms",    "越小下降越快,越大越悬浮"),
        ("最长按住(ms)",  "HOLD_MAX_S",       "ms",    "单次按住的最大时长"),
        ("按住增益",      "HOLD_GAIN",        "float", "位置误差×增益=额外按住时长"),
        ("前瞻时间(s)",   "PREDICT_AHEAD",    "float", "预测未来位置的时间"),
        ("速度阻尼",      "SPEED_DAMPING",    "float", "下坠快加按住,上升快减按住"),
        ("最大距离(px)",  "MAX_FISH_BAR_DIST","int",   "鱼条距离超过视为误检"),
        ("速度平滑",      "VELOCITY_SMOOTH",  "float", "0~1, 越大越平滑"),
        ("旋转阈值(°)",   "TRACK_MIN_ANGLE",  "float", "轨道倾斜超过此角度启用旋转"),
        ("旋转上限(°)",   "TRACK_MAX_ANGLE",  "float", "超过此角度视为误检(如海平线)"),
        ("搜索上(px)",    "REGION_UP",        "int",   "白条锁定后向上搜索的像素数"),
        ("搜索下(px)",    "REGION_DOWN",      "int",   "白条锁定后向下搜索的像素数"),
        ("搜索X(px)",     "REGION_X",         "int",   "白条中心左右各N像素范围内检测"),
        ("归正时间(s)",   "POST_CATCH_DELAY", "float", "钓鱼结束/失败后等待N秒再抛竿"),
        ("摇头时长(s)",   "SHAKE_HEAD_TIME",  "float", "摇头每段按住时长,0=不摇头"),
        ("按压时间(s)",   "INITIAL_PRESS_TIME","float", "开局按压时长(开局延迟0.5s固定)"),
        ("确认帧数",      "VERIFY_CONSECUTIVE","int",   "连续几帧检测到UI才确认小游戏开始"),
        ("成功阈值(%)",   "SUCCESS_PROGRESS", "pct",   "进度条超过此百分比判定钓鱼成功"),
    ],
    "en": [
        ("Hook Wait(s)",   "BITE_FORCE_HOOK",  "float", "Wait time from cast to bites"),
        ("Fish Size(px)",    "FISH_GAME_SIZE",   "int",   "Approximate fish icon size in game"),
        ("Dead Zone(px)",      "DEAD_ZONE",        "int",   "Larger = easier to trigger hold"),
        ("Min Hold(ms)","HOLD_MIN_S",       "ms",    "Smaller = fall faster"),
        ("Max Hold(ms)",  "HOLD_MAX_S",       "ms",    "Max hold duration per press"),
        ("Hold Gain",      "HOLD_GAIN",        "float", "Position error × gain = extra hold"),
        ("Predict(s)",   "PREDICT_AHEAD",    "float", "Prediction time for future position"),
        ("Speed Damping",      "SPEED_DAMPING",    "float", "Add hold when falling fast"),
        ("Max Dist(px)",  "MAX_FISH_BAR_DIST","int",   "Fish-bar distance threshold"),
        ("Vel Smooth",      "VELOCITY_SMOOTH",  "float", "0~1, larger = smoother"),
        ("Rot Threshold(°)",   "TRACK_MIN_ANGLE",  "float", "Enable rotation above this angle"),
        ("Rot Max(°)",   "TRACK_MAX_ANGLE",  "float", "Max reasonable angle"),
        ("Search Up(px)",    "REGION_UP",        "int",   "Pixels to search upward"),
        ("Search Down(px)",    "REGION_DOWN",      "int",   "Pixels to search downward"),
        ("Search X(px)",     "REGION_X",         "int",   "Horizontal search range"),
        ("Post Delay(s)",   "POST_CATCH_DELAY", "float", "Wait N seconds after catch/fail"),
        ("Shake Time(s)",   "SHAKE_HEAD_TIME",  "float", "Shake duration per segment, 0=off"),
        ("Init Press(s)",   "INITIAL_PRESS_TIME","float", "Initial press duration"),
        ("Verify Frames",      "VERIFY_CONSECUTIVE","int",   "Frames to confirm minigame"),
        ("Success(%)",   "SUCCESS_PROGRESS", "pct",   "Progress threshold for success"),
    ],
    "jp": [
        ("フック待機(s)",   "BITE_FORCE_HOOK",  "float", "投竿後ヒットまでの待機時間"),
        ("魚サイズ(px)",    "FISH_GAME_SIZE",   "int",   "ゲーム内魚アイコンの概算サイズ"),
        ("デッドゾーン(px)",      "DEAD_ZONE",        "int",   "大きいほどホールドしやすい"),
        ("最小ホールド(ms)","HOLD_MIN_S",       "ms",    "小さいほど速く落下"),
        ("最大ホールド(ms)",  "HOLD_MAX_S",       "ms",    "1回の最大ホールド時間"),
        ("ホールドゲイン",      "HOLD_GAIN",        "float", "位置誤差×ゲイン=追加ホールド"),
        ("予測時間(s)",   "PREDICT_AHEAD",    "float", "将来位置の予測時間"),
        ("速度ダンピング",      "SPEED_DAMPING",    "float", "高速落下時にホールド追加"),
        ("最大距離(px)",  "MAX_FISH_BAR_DIST","int",   "魚-バー距離しきい値"),
        ("速度平滑",      "VELOCITY_SMOOTH",  "float", "0~1, 大きいほど滑らか"),
        ("回転しきい値(°)",   "TRACK_MIN_ANGLE",  "float", "この角度以上で回転有効"),
        ("回転上限(°)",   "TRACK_MAX_ANGLE",  "float", "最大合理角度"),
        ("上検索(px)",    "REGION_UP",        "int",   "上方向検索ピクセル数"),
        ("下検索(px)",    "REGION_DOWN",      "int",   "下方向検索ピクセル数"),
        ("X検索(px)",     "REGION_X",         "int",   "水平検索範囲"),
        ("後処理待機(s)",   "POST_CATCH_DELAY", "float", "キャッチ/失敗後の待機時間"),
        ("振り時間(s)",   "SHAKE_HEAD_TIME",  "float", "1段の振り時間, 0=オフ"),
        ("初期押下(s)",   "INITIAL_PRESS_TIME","float", "初期押下時間"),
        ("確認フレーム",      "VERIFY_CONSECUTIVE","int",   "ミニゲーム確認フレーム数"),
        ("成功しきい値(%)",   "SUCCESS_PROGRESS", "pct",   "成功判定進行しきい値"),
    ],
}


def get_tunable_params(lang: str = None) -> list:
    """获取可调参数列表（按语言）"""
    if lang is None:
        lang = _i18n.get_language()
    return TUNABLE_PARAMS_I18N.get(lang, TUNABLE_PARAMS_I18N[DEFAULT_LANG])
