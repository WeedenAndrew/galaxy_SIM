"""Build the initial star field.

Structure of arrays, not an array of objects. A `Star` class with 40,000
instances means 40,000 Python objects and a 40,000-iteration loop per frame;
these arrays let one NumPy call move the whole galaxy.
"""

import numpy as np
import config as C
import physics


def _sample_radii(n, rng):
    """Exponential disc profile by inverse-transform sampling, truncated.

    Surface density falls as exp(-r/scale), so the mass per annulus goes as
    r*exp(-r/scale) and the CDF inverts through the Lambert W branch. Rejection
    sampling is simpler and fast enough at this size.
    """
    out = np.empty(0)
    while out.size < n:
        # Envelope: sample generously, keep by the r*exp(-r/rs) weight.
        r = rng.uniform(0.0, C.MAX_R, size=int((n - out.size) * 2.5) + 64)
        w = (r / C.DISC_SCALE_R) * np.exp(-r / C.DISC_SCALE_R)
        keep = r[rng.uniform(0.0, np.exp(-1.0), size=r.size) < w]
        out = np.concatenate([out, keep])
    return out[:n]


def _colour_from_temperature(temp):
    """Approximate blackbody colour, normalised to peak 1.0 per channel.

    Cool stars run red, hot stars blue-white. The original picked all three
    channels at random in [150, 255], which produces pastel confetti rather
    than a stellar population.
    """
    t = (temp - C.TEMP_MIN) / (C.TEMP_MAX - C.TEMP_MIN)
    t = np.clip(t, 0.0, 1.0)
    r = 1.0 - 0.42 * t
    g = 0.62 + 0.30 * t - 0.22 * t * t
    b = 0.38 + 0.62 * t
    return np.stack([r, g, b], axis=1)


def build(rng=None):
    """Return (pos, vel, colour, brightness) for the whole galaxy."""
    rng = rng or np.random.default_rng()
    n = C.STAR_COUNT
    n_bulge = int(n * 0.18)
    n_disc = n - n_bulge

    # ── disc ────────────────────────────────────────────────────────────────
    r_disc = _sample_radii(n_disc, rng)

    # Arms. A fraction of stars sit near an arm ridge; the rest are smooth
    # background. Without the background the galaxy looks like drawn spokes.
    theta = rng.uniform(0.0, 2 * np.pi, n_disc)
    on_arm = rng.random(n_disc) < C.ARM_FRACTION
    k = rng.integers(0, C.ARM_COUNT, n_disc)
    ridge = physics.arm_phase(np.maximum(r_disc, C.ARM_R0)) + 2 * np.pi * k / C.ARM_COUNT
    # Spread grows outward — arms are crisp in the inner disc and fray at the
    # edge, which is what real spirals do.
    spread = C.ARM_SPREAD * (1.0 + 1.5 * r_disc / C.MAX_R)
    theta = np.where(on_arm, ridge + rng.normal(0.0, spread, n_disc), theta)

    z_disc = rng.normal(0.0, C.DISC_THICK, n_disc)

    # ── bulge: Plummer sphere, isotropic ────────────────────────────────────
    # Conditional Plummer CDF: preserve the radial profile without a shell at MAX_R.
    # Consume the same random draws so subsequent disc randomness is unchanged.
    u_max = 1.0 / (1.0 + (C.BULGE_R / C.MAX_R) ** 2) ** 1.5
    u = rng.random(n_bulge) * u_max
    u23 = u ** (2.0 / 3.0)
    r_b = C.BULGE_R * np.sqrt(u23 / (1.0 - u23))  # also finite at u=0
    cos_i = rng.uniform(-1.0, 1.0, n_bulge)
    phi = rng.uniform(0.0, 2 * np.pi, n_bulge)
    sin_i = np.sqrt(1.0 - cos_i**2)

    pos = np.empty((n, 3))
    pos[:n_disc, 0] = r_disc * np.cos(theta)
    pos[:n_disc, 1] = r_disc * np.sin(theta)
    pos[:n_disc, 2] = z_disc
    pos[n_disc:, 0] = r_b * sin_i * np.cos(phi)
    pos[n_disc:, 1] = r_b * sin_i * np.sin(phi)
    pos[n_disc:, 2] = r_b * cos_i

    # ── velocities ──────────────────────────────────────────────────────────
    # Circular speed from the same rotation curve physics.py integrates. The
    # original drew velocities from uniform(-5000, 5000) with no reference to
    # radius at all, which is not an orbit — it is a cloud of debris.
    r = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
    r_safe = np.maximum(r, C.SOFTENING)
    v_circ = physics.circular_speed(r_safe)

    vel = np.empty((n, 3))
    vel[:, 0] = -pos[:, 1] / r_safe * v_circ
    vel[:, 1] = pos[:, 0] / r_safe * v_circ
    vel[:, 2] = 0.0

    # Velocity dispersion: real discs are not perfectly circular, and a little
    # scatter is what lets the spiral potential gather stars into arms rather
    # than sliding them past.
    #
    # In-plane only. Applying the same 6% vertically gives stars far more
    # vertical energy than the thickness they were placed with, and the disc
    # then puffs — measured at 8 px growing past 16 px, on its way to a sphere,
    # at which point TILT stops meaning anything.
    vel[:, :2] += rng.normal(0.0, 0.06, (n, 2)) * v_circ[:, None]

    # Vertical dispersion matched to the scale height instead. A star oscillates
    # about the midplane at frequency nu = sqrt(G M_enc / r^3), so the velocity
    # consistent with an amplitude of DISC_THICK is nu * DISC_THICK. Drawn from
    # the same mass model the force uses, so the disc starts in equilibrium
    # rather than relaxing into one.
    nu = np.sqrt(C.G * physics.enclosed_mass(r_safe) / r_safe**3)
    vel[:, 2] += rng.normal(0.0, 1.0, n) * nu * C.DISC_THICK

    # ── appearance ──────────────────────────────────────────────────────────
    # Young blue stars concentrate in arms; the bulge is old and red.
    temp = rng.uniform(C.TEMP_MIN, C.TEMP_MAX, n)
    temp[:n_disc] = np.where(
        on_arm,
        rng.uniform(6500, C.TEMP_MAX, n_disc),
        rng.uniform(C.TEMP_MIN, 7500, n_disc),
    )
    temp[n_disc:] = rng.uniform(C.TEMP_MIN, 5000, n_bulge)

    colour = _colour_from_temperature(temp)
    brightness = rng.uniform(0.35, 1.0, n) ** 2  # few bright, many faint
    return pos, vel, colour, brightness


def rebalance_velocities(pos, vel, pm, star_mass, rng=None):
    """Reset circular speeds to match the force the *mesh* actually delivers.

    The analytic curve and the particle mesh do not agree — measured at 11-17%,
    because a 32-cell grid smooths a concentrated bulge and because the stars
    now carry the disc mass themselves. Launching with analytic velocities
    means every star is mis-set for the force it feels, and the disc spends its
    first rotation contracting into equilibrium.

    Measuring the field and matching it removes that transient entirely, and it
    is the only way to start balanced without hand-tuning masses against a grid
    size. Circular balance is v^2 / r = -a_r, so v = sqrt(-a_r * r).
    """
    rng = rng or np.random.default_rng()

    # Route through physics.acceleration rather than summing the terms here.
    #
    # Calling `pm.accelerate() + halo_acceleration()` directly duplicates the
    # force model, and a duplicate drifts: any term added to the real force —
    # a central black hole, say — would be felt by the stars but absent from
    # the velocities they launch with, so the disc would collapse inward on
    # frame one for no visible reason. One definition of the force, used by
    # both the integrator and the initial conditions.
    acc = physics.acceleration(pos, 0.0, pm, star_mass)

    r = np.hypot(pos[:, 0], pos[:, 1])
    r_safe = np.maximum(r, C.SOFTENING)
    a_r = (acc[:, 0] * pos[:, 0] + acc[:, 1] * pos[:, 1]) / r_safe

    # Outward-pointing net force would give an imaginary speed; clamp to zero
    # and let those few stars fall inward, which is what they should do.
    v_circ = np.sqrt(np.maximum(-a_r * r_safe, 0.0))

    vel[:, 0] = -pos[:, 1] / r_safe * v_circ
    vel[:, 1] = pos[:, 0] / r_safe * v_circ
    vel[:, 2] = 0.0
    vel[:, :2] += rng.normal(0.0, 0.06, (pos.shape[0], 2)) * v_circ[:, None]

    nu = np.sqrt(np.abs(a_r) / r_safe)
    vel[:, 2] += rng.normal(0.0, 1.0, pos.shape[0]) * nu * C.DISC_THICK
    # Preserve every original draw above, keeping disc velocities bit-identical.
    # Only the bulge is overwritten, once, using measured Jeans moments.
    import jeans
    jeans.assign(pos, vel, pm, star_mass, rng)
    return vel
