"""Forces and integration. Everything here operates on whole arrays at once.

The central black hole is exaggerated for legibility. Negative mesh polarity
is a nonphysical demonstration; the halo and black hole remain attractive.

Two things the original could not have produced, and both matter for whether
it looks like a galaxy:

1. A point mass gives a Keplerian rotation curve. The outer disc then orbits
   far too slowly, differential rotation shreds any spiral within two or three
   turns, and you get a smooth ring. Real galaxies have flat curves, so the
   mass model here is bulge + disc + halo.

2. Arms are not a material structure. If stars *were* the arms, the arms would
   wind up and vanish — this is the winding problem, and it is why a purely
   kinematic spiral fails. Arms are modelled as a rigidly rotating potential
   perturbation that stars pass through.
"""

import numpy as np
import config as C


def arm_phase(r):
    """Angle of an arm's ridge line at radius r, in radians.

    A logarithmic spiral: theta = ARM_TWIST * ln(r / ARM_R0). Equivalent to a
    constant pitch angle of arctan(1 / ARM_TWIST) — about 29 degrees at 1.8,
    which is a loose, open spiral. Larger ARM_TWIST winds it tighter.

    Generation and dynamics both call this, so the arms stars are *placed* on
    and the arms the potential *pulls* toward cannot drift apart.
    """
    return C.ARM_TWIST * np.log(r / C.ARM_R0)


def enclosed_mass(r):
    """Mass inside radius r, in kg. Plummer bulge plus exponential disc.

    The disc term is the spherical approximation to an exponential profile —
    the exact result needs Bessel functions and is not visually distinguishable
    here.
    """
    # s**1.5 written as s*sqrt(s): a fractional power call is several times the
    # cost of a sqrt, and this runs over every star every substep.
    s = r * r + C.BULGE_R * C.BULGE_R
    bulge = C.BULGE_MASS * r * r * r / (s * np.sqrt(s))
    x = r / C.DISC_SCALE_R
    disc = C.DISC_MASS * (1.0 - (1.0 + x) * np.exp(-x))
    return bulge + disc


def circular_speed_squared(r):
    """v_circ(r)^2 in m^2/s^2.

    Squared, because the acceleration needs v^2/r and taking a square root only
    to square it again costs a full sqrt over every star, twenty times a frame.
    """
    r = np.maximum(r, 1e-6)
    v2_grav = C.G * enclosed_mass(r) / r
    v2_halo = C.HALO_V_FLAT**2 * r**2 / (r**2 + C.HALO_CORE_R**2)
    return v2_grav + v2_halo


def circular_speed(r):
    """Circular orbital speed at radius r, in m/s. Used at setup, not per frame.

    The halo term is a pseudo-isothermal profile, flattening to HALO_V_FLAT at
    large radius — that flattening is the whole reason the halo is here.
    """
    return np.sqrt(circular_speed_squared(r))


def halo_acceleration(pos):
    """Static dark-matter halo only. The stars supply everything else.

    Standard practice in N-body galaxy work: a live disc inside a rigid halo.
    A disc of visible matter alone cannot hold a flat rotation curve — that is
    the observation dark matter was invented to explain — so leaving the halo
    out gives a Keplerian falloff no matter how good the star-star gravity is.
    """
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    r2 = x * x + y * y + z * z
    r2_safe = np.maximum(r2, C.SOFTENING**2)
    # Pseudo-isothermal: a = -v_flat^2 * r / (r^2 + rc^2), flattening at large r.
    k = -C.HALO_V_FLAT**2 / (r2_safe + C.HALO_CORE_R**2)
    acc = np.empty_like(pos)
    acc[:, 0] = k * x
    acc[:, 1] = k * y
    acc[:, 2] = k * z
    return acc


def black_hole_acceleration(pos):
    """Plummer-softened central point mass, including a finite force at zero."""
    r2 = np.einsum('ij,ij->i', pos, pos)
    s = r2 + C.BLACK_HOLE_SOFTENING**2
    k = -C.G * C.BLACK_HOLE_MASS / (s * np.sqrt(s))
    return pos * k[:, None]


def acceleration(pos, t, pm=None, star_mass=0.0, polarity=1.0, *, sources=None):
    """Acceleration on every star. pos is (N,3); returns (N,3).

    `sources` supplies a separate mass distribution for setup-only probes.
    Without it, positions are both the sources and evaluation points.

    With `pm` supplied, the bulge and disc are not analytic terms — they are
    whatever the stars have actually arranged themselves into, felt through the
    particle mesh. That is the difference between an animation of a galaxy and
    a simulation of one: nothing here imposes the shape, so the shape is free
    to change, and structure that forms was not put there.
    """
    if pm is not None:
        acc = (pm.accelerate(pos, star_mass) if sources is None
               else pm.accelerate_at(sources, star_mass, pos))
        if polarity != 1.0:
            acc *= polarity
        acc += halo_acceleration(pos)
    else:
        # Fallback: fully analytic, every star on rails. Kept because it is a
        # useful control — if something odd appears with gravity on, run it
        # here to see whether the mesh caused it.
        x, y = pos[:, 0], pos[:, 1]
        r = np.sqrt(x * x + y * y)
        r_safe = np.maximum(r, C.SOFTENING)
        a_r = -circular_speed_squared(r_safe) / r_safe
        inv_r = 1.0 / r_safe
        acc = np.empty_like(pos)
        acc[:, 0] = a_r * x * inv_r
        acc[:, 1] = a_r * y * inv_r
        acc[:, 2] = (
            -C.G * enclosed_mass(r_safe) * pos[:, 2]
            / (r_safe**2 + pos[:, 2] ** 2 + C.SOFTENING**2) ** 1.5
        )
        # Polarity deliberately does nothing on this path. The point of rails
        # mode is to be the unchanged reference you compare against when the
        # mesh does something surprising.
        if C.BLACK_HOLE_MASS:
            acc += black_hole_acceleration(pos)

    # Optional imposed spiral, off by default now that gravity is live.
    if C.SPIRAL_STRENGTH:
        x, y = pos[:, 0], pos[:, 1]
        r = np.sqrt(x * x + y * y)
        r_safe = np.maximum(r, C.SOFTENING)
        phase = C.ARM_COUNT * (np.arctan2(y, x) - arm_phase(r_safe) - C.OMEGA_P * t)
        a_s = C.SPIRAL_STRENGTH * np.cos(phase) * np.exp(-r / C.DISC_SCALE_R)
        acc[:, 0] += a_s * x / r_safe
        acc[:, 1] += a_s * y / r_safe
    return acc


def step(pos, vel, dt, t, acc, pm=None, star_mass=0.0, polarity=1.0):
    """One kick-drift-kick leapfrog step, in place. Returns (t, acc).

    Leapfrog rather than the explicit Euler the original used. Euler adds energy
    every step: orbits spiral outward and the galaxy visibly inflates over a few
    thousand frames. Leapfrog is symplectic, so the error oscillates instead of
    accumulating.

    `acc` is carried across calls. The closing kick of one step and the opening
    kick of the next use the same acceleration, so evaluating it twice does
    identical work twice — which was half the frame budget.
    """
    vel += 0.5 * dt * acc
    pos += dt * vel
    acc = acceleration(pos, t + dt, pm, star_mass, polarity)
    vel += 0.5 * dt * acc
    return t + dt, acc
