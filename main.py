"""
VRChat 自動釣りスクリプト — エントリーポイント
============================
tkinter GUI を起動する。

ショートカットキー（VRChat内でも有効）:
    F9  = 開始 / 一時停止
    F10 = 停止
    F11 = デバッグモード
"""

import tkinter as tk
from gui.app import FishingApp


def main():
    root = tk.Tk()
    FishingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()