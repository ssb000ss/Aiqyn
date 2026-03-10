# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Aiqyn — single-directory bundle (onedir)."""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect spaCy model data
datas = []
datas += collect_data_files("spacy")
datas += collect_data_files("ru_core_news_sm")
datas += collect_data_files("razdel")
datas += collect_data_files("structlog")

# Include project data files
datas += [
    ("data", "data"),
    ("config", "config"),
]

# Hidden imports — dynamically loaded modules
hiddenimports = []
hiddenimports += collect_submodules("spacy")
hiddenimports += collect_submodules("ru_core_news_sm")
hiddenimports += collect_submodules("aiqyn.extractors")
hiddenimports += [
    "aiqyn",
    "aiqyn.cli.main",
    "aiqyn.core.analyzer",
    "aiqyn.core.pipeline",
    "aiqyn.core.aggregator",
    "aiqyn.core.segmenter",
    "aiqyn.core.preprocessor",
    "aiqyn.core.calibrator",
    "aiqyn.extractors.registry",
    "aiqyn.models.manager",
    "aiqyn.models.ollama_runner",
    "aiqyn.storage.database",
    "aiqyn.reports.pdf_exporter",
    "aiqyn.ui.app",
    "razdel",
    "pydantic",
    "pydantic_settings",
    "structlog",
    "httpx",
    "reportlab",
    "reportlab.lib.styles",
    "reportlab.platypus",
    "sqlite3",
    # PySide6 plugins
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PySide6.QtPrintSupport",
]

a = Analysis(
    ["src/aiqyn/__main__.py"],
    pathex=["src"],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=["installer/hooks"],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "notebook",
        "IPython",
        "pandas",
        "sklearn",
        "torch",
        "tensorflow",
        "cv2",
        "PIL",
        "tkinter",
        "_tkinter",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="aiqyn",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # no console window on Windows
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="installer/aiqyn.ico" if Path("installer/aiqyn.ico").exists() else None,
    version="installer/version_info.txt" if Path("installer/version_info.txt").exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="aiqyn",
)
