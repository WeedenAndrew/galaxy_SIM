"""Particle-Mesh self-gravity: every star pulls on every other star.

Direct summation is O(N^2) — 1.6 billion pair forces at 40,000 stars. Barnes-Hut
is O(N log N) but spends its time building an octree, and tree building in Python
is interpreted work NumPy cannot vectorise away.

PM sidesteps both. Smear the stars onto a grid, solve Poisson in Fourier space
where it becomes a division, interpolate the forces back. Cost is O(N) for
deposition plus O(M log M) for the FFT, independent of clustering, and every
step is one NumPy call.

    laplacian(phi) = 4 pi G rho     ->     phi_k = -4 pi G rho_k / k^2

**Isolated boundaries.** An FFT is periodic, so the galaxy would feel copies of
itself in every direction. Depositing into a box twice as wide in each axis
pushes those images far enough that their pull is negligible. Costs 8x the
cells, and it is the difference between a galaxy and one being tidally torn by
its own reflections.

**Why two FFTs and not four.** The obvious route takes the gradient in Fourier
space, which needs an inverse transform per axis — four transforms per step,
measured at 20 ms and hopeless. Solving for the potential once and
finite-differencing it on the grid costs two transforms and a few array slices.
"""

import numpy as np
import config as C

# SciPy's pocketfft is roughly 4x faster than NumPy's for this transform —
# measured 6.05 ms against 1.51 ms on a 64^3 complex round trip, identical
# results to within float32 precision. Since the FFT is most of the gravity
# budget, that is the largest single win available here and it costs an import.
#
# Optional, because the simulation should still run on a bare NumPy install.
try:
    import scipy.fft as _sfft

    def _rfftn(a):
        return _sfft.rfftn(a, workers=C.FFT_WORKERS)

    def _irfftn(a, s):
        return _sfft.irfftn(a, s=s, workers=C.FFT_WORKERS)

    FFT_BACKEND = "scipy"
except ImportError:  # pragma: no cover
    def _rfftn(a):
        return np.fft.rfftn(a)

    def _irfftn(a, s):
        return np.fft.irfftn(a, s=s, axes=(0, 1, 2))

    FFT_BACKEND = "numpy"


class PMGravity:
    def __init__(self, grid=C.PM_GRID, half_width=C.PM_HALF_WIDTH):
        self.n = grid
        self.n2 = grid * 2                      # padded box
        self.half = half_width
        self.cell = 2.0 * half_width / grid

        kx = np.fft.fftfreq(self.n2, d=self.cell) * 2.0 * np.pi
        kz = np.fft.rfftfreq(self.n2, d=self.cell) * 2.0 * np.pi
        k2 = kx[:, None, None] ** 2 + kx[None, :, None] ** 2 + kz[None, None, :] ** 2
        k2[0, 0, 0] = 1.0                       # avoid 0/0
        self.green = (-4.0 * np.pi * C.G / k2).astype(np.complex64)
        self.green[0, 0, 0] = 0.0               # drop the mean mode: no net offset

        self._rho = np.zeros((self.n2, self.n2, self.n2), dtype=np.float32)
        self._acc = np.zeros((0, 3))
        self._green_scaled = None
        self._scaled_for = None
        self.escaped = 0

    def _weights(self, pos):
        """Cloud-in-cell: a star's mass splits across the 8 cells it lies between.

        Nearest-grid-point is shorter and gives forces that jump as stars cross
        cell boundaries. CIC is the cheapest scheme with continuous forces.
        """
        g = (pos + self.half) / self.cell + self.n * 0.5

        # A star can be flung clean out of the box. Its position stays finite —
        # just enormous — and `np.floor(1e300).astype(np.int32)` is undefined:
        # it returns INT32_MIN. Clipping *after* the cast then maps that to
        # cell 0, so an escaped star dumps its entire mass into a corner of the
        # grid and warps the potential for every other star. The index
        # arithmetic overflows on the way, and the whole field goes NaN.
        #
        # Clip in float space, before the cast, where clipping still means
        # something.
        finite = np.isfinite(g)
        inside = np.all(finite, axis=1) & np.all(
            (g >= 0.0) & (g <= self.n2 - 2), axis=1
        )
        # NaN must be replaced, not clipped. `np.clip` propagates NaN — it is a
        # comparison, and every comparison against NaN is false — so clipping
        # alone leaves it to reach the cast and produce INT32_MIN anyway.
        # Infinity does clamp correctly; NaN is the case that needs this.
        g = np.where(finite, g, 0.0)
        np.clip(g, 0.0, float(self.n2 - 2), out=g)

        i0 = np.floor(g).astype(np.int32)
        f = (g - i0).astype(np.float32)
        ny = self.n2
        base = (i0[:, 0] * ny + i0[:, 1]) * ny + i0[:, 2]

        # Escaped stars carry zero weight: they deposit no mass and receive no
        # mesh force. The analytic halo still pulls on them, which is both
        # correct — they are outside the region the mesh describes — and useful,
        # since it eventually brings them home.
        #
        # Both complements are computed once rather than four times each, and
        # the escape mask is folded into the x factors rather than multiplied
        # into all eight corners. The naive loop does 12 redundant subtractions
        # and 8 redundant masks over 40,000 elements — measured at a third of
        # the whole particle-mesh budget, for arithmetic that never changes.
        w_in = inside.astype(np.float32)
        fx1, fy1, fz1 = f[:, 0], f[:, 1], f[:, 2]
        fx0, fy0, fz0 = 1.0 - fx1, 1.0 - fy1, 1.0 - fz1
        fx0 = fx0 * w_in
        fx1 = fx1 * w_in

        corners = []
        for dx, wx in ((0, fx0), (1, fx1)):
            for dy, wy in ((0, fy0), (1, fy1)):
                wxy = wx * wy
                for dz, wz in ((0, fz0), (1, fz1)):
                    corners.append(
                        (base + (dx * ny + dy) * ny + dz, wxy * wz)
                    )
        self.escaped = int((~inside).sum())
        return corners

    def accelerate(self, pos, star_mass):
        """(N,3) acceleration from the stars' own gravity."""
        corners = self._weights(pos)

        # Deposit dimensionless *counts*, then scale to density afterwards.
        #
        # Depositing `star_mass * w` directly puts ~3e36 kg into every cell,
        # and a crowded bulge cell holding a thousand stars reaches 3e39 —
        # past the float32 ceiling of 3.4e38. The grid silently becomes `inf`,
        # the FFT propagates it everywhere, and every star's acceleration is
        # NaN from the first call. Nothing errors; the galaxy just stops
        # existing.
        self._rho.fill(0.0)
        flat = self._rho.ravel()
        for idx, w in corners:
            np.add.at(flat, idx, w)

        # Fold the density scale into the Green's function BEFORE it touches
        # the transformed density.
        #
        # `rfftn(rho) * green * rho_scale` evaluates left to right, so it forms
        # `rho_k * green` first — of order 3.6e36 — and only then multiplies by
        # rho_scale (~9e-23) to bring it back. That intermediate sits just under
        # the complex64 ceiling of 3.4e38, so a crowded frame overflows to inf,
        # the inverse transform spreads NaN across the grid, and every star's
        # position follows. The visible symptom is a blit error about surface
        # dimensions, three layers away from the cause.
        #
        # Scaling green first keeps the largest intermediate around 8e9.
        if star_mass != self._scaled_for:
            self._green_scaled = self.green * (star_mass / self.cell**3)
            self._scaled_for = star_mass

        phi = _irfftn(_rfftn(self._rho) * self._green_scaled, (self.n2,) * 3)

        # Central differences in place. np.gradient is clearer and allocates
        # three full grids per call, which measured as a third of the budget.
        inv2h = 1.0 / (2.0 * self.cell)
        if self._acc.shape[0] != pos.shape[0]:
            self._acc = np.zeros((pos.shape[0], 3))

        g_field = np.zeros_like(phi)
        for axis in range(3):
            g_field.fill(0.0)
            a = [slice(None)] * 3
            b = [slice(None)] * 3
            o = [slice(None)] * 3
            a[axis], b[axis], o[axis] = slice(2, None), slice(0, -2), slice(1, -1)
            g_field[tuple(o)] = (phi[tuple(a)] - phi[tuple(b)]) * inv2h

            # Interpolate back with the *same* CIC weights used to deposit.
            # A different scheme here makes each star feel its own mass and the
            # disc heats until it evaporates.
            acc = self._acc[:, axis]
            acc.fill(0.0)
            gf = g_field.ravel()
            for idx, w in corners:
                acc -= gf[idx] * w
        return self._acc
