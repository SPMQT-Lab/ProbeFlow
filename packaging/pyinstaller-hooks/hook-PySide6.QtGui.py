"""Collect only the Qt GUI plugins ProbeFlow can use.

PyInstaller's generic QtGui hook collects every plugin in each supported
plugin category. The PySide6 wheels consequently pull Qt Virtual Keyboard
(GPL-only for open-source users) and the Qt PDF image plugin into an ordinary
Qt Widgets application that imports neither component.
"""

from pathlib import Path

from PyInstaller.utils.hooks.qt import pyside6_library_info


hiddenimports, binaries, datas = pyside6_library_info.collect_module("PySide6.QtGui")

_UNUSED_PLUGIN_FILENAMES = {
    "libqpdf.dylib",
    "libqtvirtualkeyboardplugin.dylib",
    "qpdf.dll",
    "qtvirtualkeyboardplugin.dll",
}
binaries = [
    (source, destination)
    for source, destination in binaries
    if Path(source).name.lower() not in _UNUSED_PLUGIN_FILENAMES
]
