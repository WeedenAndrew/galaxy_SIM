"""Setup-only spherical isotropic Jeans moments in the measured total field.

No escape-energy reference or potential offset is needed. A Gaussian matched
to these moments is a defensible Jeans approximation, not an exact Plummer DF
or a proof of equilibrium in the nonspherical disc+mesh field. The tracer
tracer integral ends at MAX_R, matching the truncated position sample and
imposing zero radial pressure at that boundary.
"""
import numpy as np
import config as C
import physics


def sphere_directions(count):
    """Antipodal equal-area Fibonacci directions; no preferred mean vector."""
    if count < 4 or count % 2:
        raise ValueError('direction count must be even and at least four')
    i = np.arange(count // 2)
    z = (i + 0.5) / (count // 2)
    azimuth = i * (np.pi * (3.0 - np.sqrt(5.0)))
    xy = np.sqrt(1.0 - z*z)
    half = np.column_stack((xy*np.cos(azimuth), xy*np.sin(azimuth), z))
    return np.concatenate((half, -half))


def profile(sources, pm, star_mass, *, directions=64, radial_points=512,
            outer_radius=None):
    """Return radii (m), sigma (m/s), mean inward g (m/s^2).

    Defaults are setup quadrature choices, not tunable physical parameters.
    The default tracer boundary is MAX_R with zero radial pressure there.
    Density normalization cancels from Jeans. An explicit outer_radius is
    available for diagnostic comparisons, not used by bulge assignment.
    """
    outer_radius = C.MAX_R if outer_radius is None else outer_radius
    radius = np.geomspace(0.1*C.MPP, outer_radius, radial_points)
    unit = sphere_directions(directions)
    probes = (radius[:,None,None] * unit[None,:,:]).reshape(-1,3)
    acc = physics.acceleration(probes, 0.0, pm, star_mass, sources=sources)
    inward = -np.einsum('rdi,di->rd', acc.reshape(-1,directions,3), unit).mean(axis=1)
    if not np.isfinite(inward).all() or np.any(inward < 0):
        raise ValueError('Jeans requires a finite, inward spherical mean force')
    rho = (1.0 + (radius/C.BULGE_R)**2)**(-2.5)
    integrand = rho * inward
    trapezoids = 0.5*(integrand[:-1]+integrand[1:])*np.diff(radius)
    pressure = np.zeros(radial_points)
    pressure[:-1] = np.cumsum(trapezoids[::-1])[::-1]
    sigma = np.sqrt(pressure/rho)
    return radius, sigma, inward


def assign(pos, vel, pm, star_mass, rng):
    """Replace only the last 18% with independent Gaussian components."""
    count = int(len(pos)*0.18)
    if count == 0:
        return
    radius, sigma, _ = profile(pos, pm, star_mass)
    bulge_r = np.linalg.norm(pos[-count:], axis=1)
    local_sigma = np.interp(bulge_r, radius, sigma)
    # Three independent N(0,sigma) components give isotropic directions without
    # uniform-angle pole bias. No rotation term, escape cutoff or fudge factor.
    vel[-count:] = rng.normal(size=(count,3))*local_sigma[:,None]
