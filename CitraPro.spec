# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
import sys

sys.setrecursionlimit(sys.getrecursionlimit() * 5)

block_cipher = None

# --- Pengumpulan Data Library (Bagian Penting) ---
datas = []
datas += collect_data_files('cv2')
datas += collect_data_files('matplotlib')
datas += collect_data_files('PIL')

# --- Analisis Aplikasi ---
a = Analysis(
    ['Pencit.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'PIL._tkinter_finder',
        'matplotlib.backends.backend_tkagg',
        'cv2.data'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# --- Konfigurasi Executable ---
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CitraPro',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False, # Set False untuk menghilangkan jendela hitam CMD
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# --- Pengumpulan Final ---
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CitraPro', # Nama folder output di dalam 'dist'
)