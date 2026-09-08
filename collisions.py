"""Decorative particle proximity events, not physical stellar collisions.

Real stellar collisions are vanishingly rare: disc stars are separated by
millions of stellar diameters. Each particle here represents 2.4 million solar
masses at 25,000 particles (1.5 million at 40,000), so close particles do not
mean two stars touched. These events are for flashes only; never change mass,
position or velocity. Reversed gravity is nonphysical, and the exaggerated
black hole is a choice about legibility rather than a measurement.

Only pairs in the SAME cell are tested. We intentionally miss pairs across
cell boundaries to keep this decorative effect cheap; counts are incomplete
and depend on grid alignment. This is not a physical collision census.

Sorting/grouping costs O(N log N), plus O(K) local candidate pairs. No global
N-by-N array is built. A pathological crowded cell can still require O(N^2)
work/output; spatial hashing cannot guarantee linear cost for arbitrary input.
"""

import numpy as np
import config as C


def detect(pos, radius=None, *, return_candidates=False):
    """Return unique (i,j), i<j, within radius and the same hash cell.

    Input positions and radius are in metres. Nonfinite positions and cells
    outside [-2**20, 2**20) on any axis are excluded, not clamped into events.
    Optional candidate count supports auditing the broad-phase workload.
    """
    radius = C.COLLISION_RADIUS if radius is None else float(radius)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('radius must be finite and positive')
    pos = np.asarray(pos)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError('pos must have shape (N,3)')
    limit = 2**20
    # Division can overflow for huge finite positions or tiny radii. Such
    # coordinates are intentionally rejected below, never cast or deposited.
    with np.errstate(over='ignore', invalid='ignore'):
        q = pos / radius
    valid = np.all(np.isfinite(q) & (q >= -limit) & (q < limit), axis=1)
    ids = np.flatnonzero(valid)
    # Filter in FLOAT space before floor/cast: NaN and 1e300 cannot reach it.
    cells = np.floor(q[valid]).astype(np.int64) + limit
    # 21 bits per axis: maximum packed key is 2**63-1, no int64 overflow.
    keys = (cells[:, 0] * (2*limit) + cells[:, 1]) * (2*limit) + cells[:, 2]
    order = np.argsort(keys)
    sorted_keys = keys[order]
    boundaries = np.flatnonzero(np.diff(sorted_keys)) + 1
    starts = np.concatenate(([0], boundaries))
    counts = np.diff(np.concatenate((starts, [keys.size])))
    chunks = []
    candidates = 0
    for start, count in zip(starts[counts > 1], counts[counts > 1]):
        group = ids[order[start:start+count]]
        candidates += int(count * (count - 1) // 2)
        # Row-wise candidates avoid even a dense-cell square temporary.
        for offset in range(count - 1):
            other = group[offset+1:]
            delta = q[group[offset]] - q[other]
            hit = other[np.einsum('ij,ij->i', delta, delta) < 1.0]
            if hit.size:
                first = np.full(hit.size, group[offset], dtype=np.intp)
                chunks.append(np.column_stack((np.minimum(first,hit), np.maximum(first,hit))))
    pairs = np.concatenate(chunks) if chunks else np.empty((0,2), dtype=np.intp)
    return (pairs, candidates) if return_candidates else pairs
