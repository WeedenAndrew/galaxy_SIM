"""Every constant, with its unit stated.

The original had `sm`, `mpp` and `g` used but never defined, and `gmass`
defined but never used — gravity read `mass`, which only existed because the
star-creation loop leaked its variable to module scope. So the whole galaxy was
pulled toward whatever the last star's random mass happened to be.

That class of bug is why units are in the names here and why nothing is a bare
number below.
"""

import math

# ── physical constants, SI ──────────────────────────────────────────────────
G = 6.67430e-11          # m^3 kg^-1 s^-2
SOLAR_MASS = 1.98892e30  # kg
PARSEC = 3.0857e16       # m
KPC = 1000 * PARSEC
YEAR = 3.15576e7         # s

# ── window ──────────────────────────────────────────────────────────────────
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
TARGET_FPS = 60

# ── scale ───────────────────────────────────────────────────────────────────
# Metres per pixel. 500 px of disc radius has to fit in a 600 px half-window,
# so this is what ties physical size to what you actually see.
MPP = 30 * PARSEC        # m/px  -> ~15 kpc visible radius

# ── the galaxy ──────────────────────────────────────────────────────────────
DISC_SCALE_R = 130 * MPP   # exponential disc scale length (was `rd`)
MAX_R        = 500 * MPP   # sampling cutoff (was `max_r`)
BULGE_R      = 45 * MPP    # Plummer bulge scale radius (was `bulge_r`)
DISC_THICK   = 8 * MPP     # vertical scale height; needed for TILT to read

# 25,000 rather than 40,000. Render cost scales with this; the measured target
# machine ran 40k at ~33 fps, and dropping star count is the cheaper of the two
# levers because it costs detail you cannot see rather than force resolution
# you can.
STAR_COUNT = 25_000

# Mass model. A single point mass gives a Keplerian curve (v ~ 1/sqrt(r)),
# which makes the outer disc orbit too slowly and winds the arms into mush
# within a few rotations. Real galaxies are flat, so: bulge + disc + halo.
BULGE_MASS = 2.5e8 * SOLAR_MASS    # was `gmass`, the only mass you had
DISC_MASS  = 6.0e10 * SOLAR_MASS
HALO_V_FLAT = 220_000.0            # m/s, asymptotic circular speed
HALO_CORE_R = 200 * MPP            # halo core radius

# Visual exaggeration: 250 times the 4e6-solar-mass Sagittarius A* reference.
BLACK_HOLE_MASS = 1e9 * SOLAR_MASS  # kg; chosen for legibility, not measurement
BLACK_HOLE_SOFTENING = 5 * MPP     # 150 pc, Plummer core

SOFTENING = 25 * MPP     # Plummer softening, stops r->0 blowing up (was `eps`)

# ── spiral arms ─────────────────────────────────────────────────────────────
ARM_COUNT  = 6      # number of arms
ARM_TWIST  = 1.8    # winding: radians of arm rotation per e-fold in radius
ARM_SPREAD = 0.228  # angular sigma of stars about an arm, radians
ARM_R0     = 10 * MPP   # radius where arm phase is zero (was `spiral_r0`)
ARM_FRACTION = 0.68     # fraction of disc stars placed on arms vs. smooth

# Rotating spiral potential. This is what keeps arms from winding up: the
# pattern rotates rigidly at OMEGA_P while stars orbit at their own rate.
# Zero once self-gravity is live: an imposed pattern on top of stars that
# already pull on each other is double-counting, and it would mask whatever
# structure actually forms. Raise it to force arms the sim won't make itself.
SPIRAL_STRENGTH = 0.0       # m/s^2, perturbation amplitude
OMEGA_P = 1.5e-15           # rad/s, pattern speed

# ── self-gravity (particle mesh) ────────────────────────────────────────────
# Cells across the galaxy region. The FFT runs on a box twice this wide in each
# axis, so cost goes as (2*PM_GRID)^3 — 32 measured at ~8 ms, 40 at ~17 ms.
# That ceiling, not the star count, is what sets the resolution.
# 24, not 32. The FFT runs on (2*grid)^3 cells, so this is the steepest dial in
# the project — dropping one step cuts particle-mesh cost by roughly 30%.
#
# What it costs: the cell grows from 35 px to 47 px, which is wider than the
# 45 px bulge scale radius. The core is therefore smoothed across a single
# cell and the inner force profile is approximate. The disc is unaffected; the
# bulge is where you are paying.
PM_GRID = 24
PM_HALF_WIDTH = 560 * 30 * PARSEC   # slightly beyond MAX_R, so no star leaves the box

# Threads for the FFT, when SciPy is installed. -1 uses every core.
#
# More is not automatically faster: at 64^3 the transform is small enough that
# thread setup can cost more than it saves — measured 1.51 ms single-threaded
# against 1.74 ms on two cores. Worth trying both on your own machine, since
# the crossover depends on core count and memory bandwidth.
FFT_WORKERS = -1

# Total stellar mass, split evenly. 40k particles standing in for 6e10 solar
# masses means each is ~1.5e6 — far too heavy for real two-body encounters,
# which is exactly why PM is used: the grid smooths interactions to scales
# where the particle count stops mattering.
STELLAR_MASS = 6.0e10 * SOLAR_MASS

# The stars cannot supply a flat rotation curve on their own — a disc of only
# visible matter falls off Keplerian. The halo stays analytic, as it does in
# real N-body work: live disc, static halo.
USE_LIVE_GRAVITY = True

# ── camera ──────────────────────────────────────────────────────────────────
TILT = math.radians(55)          # starting pitch; 0 = face-on, pi/2 = edge-on
CAM_START_DISTANCE = 1400 * 30 * PARSEC
CAM_MIN_DISTANCE = 60 * 30 * PARSEC     # inside the bulge
CAM_MAX_DISTANCE = 9000 * 30 * PARSEC   # whole structure, well outside the disc
CAM_FOCAL = 1.1                  # larger = narrower field of view
CAM_NEAR = 5.0                   # px-equivalents; clips stars behind the eye
CAM_ORBIT_SPEED = 0.006          # radians per pixel of mouse drag
CAM_ZOOM_STEP = 1.12             # per scroll notch

# ── integration ─────────────────────────────────────────────────────────────
# SIM_SPEED seconds of galaxy per second of wall clock, split over SUBSTEPS.
#
# The original 1e12 with 10 substeps was wrong in both directions at once.
# An orbit at r=200px takes ~176 Myr, so 1e12 advances 3e-6 of a rotation per
# frame — 92 minutes of staring to watch the galaxy turn once. And ten substeps
# at that step size is ten times the work for no accuracy: the step was already
# a millionth of an orbit.
#
# 2e14 turns the disc once in about 25 seconds. One substep, now that each one
# costs a full Poisson solve — the step is still a ten-thousandth of an orbit,
# so a second substep buys accuracy nobody can see for half the frame rate.
SIM_SPEED = 2e14
SUBSTEPS = 1

# ── appearance ──────────────────────────────────────────────────────────────
TEMP_MIN = 3000.0    # K, coolest star
TEMP_MAX = 12000.0   # K, hottest
EXPOSURE = 0.45      # global brightness before clipping

# Decorative same-cell proximity threshold, not a stellar contact radius.
# Gate 4: +1.552 ms/frame exceeds the 0.5 ms acceptance threshold.
COLLISIONS_ENABLED = False
COLLISION_RADIUS = 0.5 * MPP  # 15 pc; spatial hash cell width

COLLISION_INTERVAL = 4  # sample decorative events every fourth simulation frame
MAX_FLASHES = 256
FLASH_LIFETIME = 0.6    # wall-clock seconds, independent of SIM_SPEED
FLASH_RISE = 0.04       # wall-clock seconds to peak
FLASH_PEAK = 4.0
