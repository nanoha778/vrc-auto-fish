"""
PyInstaller 実行時フック — パッチローダー
==========================================

exe の横に patch/ ディレクトリがある場合、
中の .py ファイルで凍結コードを上書きする。

優先順位
patch/ > exe 内のコード

例:

patch/
  config.py
  core/
    bot.py
  gui/
    app.py
"""
import sys
import os
import importlib
import importlib.util


def _setup_patch():
    if not getattr(sys, 'frozen', False):
        return

    app_dir = os.path.dirname(sys.executable)
    patch_dir = os.path.join(app_dir, 'patch')

    if not os.path.isdir(patch_dir):
        return

    class _PatchFinder:
        """在 FrozenImporter 之前拦截, 从 patch/ 加载 .py"""

        def find_spec(self, fullname, path, target=None):
            parts = fullname.split('.')

            pkg_init = os.path.join(patch_dir, *parts, '__init__.py')
            if os.path.isfile(pkg_init):
                return importlib.util.spec_from_file_location(
                    fullname, pkg_init,
                    submodule_search_locations=[
                        os.path.join(patch_dir, *parts)
                    ],
                )

            if len(parts) > 1:
                mod_file = os.path.join(
                    patch_dir, *parts[:-1], parts[-1] + '.py'
                )
            else:
                mod_file = os.path.join(patch_dir, parts[0] + '.py')

            if os.path.isfile(mod_file):
                return importlib.util.spec_from_file_location(
                    fullname, mod_file
                )

            return None

    sys.meta_path.insert(0, _PatchFinder())

    n = sum(
        1 for r, _, fs in os.walk(patch_dir)
        for f in fs if f.endswith('.py')
    )
    print(f"[补丁] 已加载 patch/ 目录 ({n} 个文件)")


_setup_patch()
