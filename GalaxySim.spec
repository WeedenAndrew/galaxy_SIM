# PyInstaller build spec.  Build with:  pyinstaller GalaxySim.spec
#
# A spec file rather than a long command line, so the build is versioned and
# repeatable instead of living in someone's shell history.

# ── the two choices that matter ──────────────────────────────────────────────

ONEFILE = False
# False -> dist/GalaxySim/GalaxySim.exe, a folder you click into. Starts
#          immediately.
# True  -> dist/GalaxySim.exe, a single file. Tidier to move or send, but every
#          launch unpacks ~200 MB of NumPy and SciPy to a temp directory first,
#          which is several seconds of nothing happening before the window
#          appears. For something you run yourself, the folder is the better
#          click.

BUNDLE_SCIPY = True
# SciPy is the largest single dependency here and buys about 1.8 ms a frame.
# `gravity.py` already falls back to NumPy's FFT without it, so setting this
# False produces a much smaller build that still runs — measured 59 fps rather
# than 63 on the reference machine. Keep it unless size actually matters.

# ─────────────────────────────────────────────────────────────────────────────

excludes = [
    # Never needed at runtime and each drags in a large tree.
    "tkinter", "matplotlib", "PIL", "IPython", "pytest",
    "pandas", "sqlite3", "unittest", "pydoc", "doctest",
]
if not BUNDLE_SCIPY:
    excludes.append("scipy")

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=[],
    # config/galaxy/gravity/physics/render/camera/collisions/flashes are all
    # reached by ordinary imports from main.py, so PyInstaller finds them.
    # bench*.py are deliberately absent: they are development tools and would
    # only add weight.
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

if ONEFILE:
    exe = EXE(
        pyz, a.scripts, a.binaries, a.datas, [],
        name="GalaxySim",
        debug=False,
        strip=False,
        upx=False,          # UPX corrupts some NumPy binaries; not worth it
        console=False,      # no terminal window — see the note below
        icon=None,
    )
else:
    exe = EXE(
        pyz, a.scripts, [],
        exclude_binaries=True,
        name="GalaxySim",
        debug=False,
        strip=False,
        upx=False,
        console=False,
        icon=None,
    )
    coll = COLLECT(
        exe, a.binaries, a.datas,
        strip=False, upx=False,
        name="GalaxySim",
    )

# console=False is what makes it a clean one-click launch, and it also means a
# crash closes the window silently. `main._run()` catches that and writes
# `galaxy-sim-crash-<timestamp>.log` beside the executable.
#
# If a build will not start at all and produces no log, the failure is before
# Python runs — set console=True, rebuild, and run it from a terminal to see
# the import error.
