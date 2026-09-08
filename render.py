"""Draw the galaxy by scattering light into a buffer, not by drawing shapes.

`pygame.draw.circle` per star is roughly 40,000 Python calls per frame, which
is about two orders of magnitude too slow here. Instead every star's screen
position is computed as one array operation and `np.add.at` accumulates
brightness into an RGB buffer that gets blitted once.

Additive accumulation also gives the effect for free: where stars crowd — the
bulge, the arm ridges — light piles up and saturates to white, which is exactly
how a galaxy looks. Drawing opaque circles cannot do that at any speed.
"""

import numpy as np
import pygame
import config as C

# Projection now lives in camera.py — it needs orbit and zoom state, which a
# module-level function had nowhere to keep.


def draw(surface, pos, colour, brightness, buffer, camera, flashes=None):
    """Accumulate all stars into `buffer` and blit it to `surface`.

    Returns (stars_drawn, buffer) — the buffer may have been reallocated.
    """
    # Size from the surface, never from config. The window is resizable, and on
    # Windows a display-scaling setting can hand back a surface that is not the
    # size that was asked for. Either way `blit_array` then raises "array must
    # match surface dimensions", which reads like a rendering bug and is really
    # a bookkeeping one.
    width, height = surface.get_size()
    if buffer.shape[0] != width or buffer.shape[1] != height:
        buffer = np.zeros((width, height, 3), dtype=np.float32)

    star_count = len(pos)
    if flashes is not None:
        active = flashes.brightness > 0.0
        pos = np.concatenate((pos, flashes.pos[active]))
        colour = np.concatenate((colour, flashes.colour[active]))
        brightness = np.concatenate((brightness, flashes.brightness[active]))
    # Stars and flashes share projection, unique/bincount accumulation and blit.
    sx, sy, depth, scale, in_front = camera.project(pos, width, height)

    # Guard the cast, not just the NaNs.
    #
    # `np.isfinite` alone is not enough: a star flung far from the camera
    # projects to a coordinate like 1e12, which is perfectly finite and still
    # casts to INT32_MIN. Testing the bounds *before* the cast, in float space,
    # covers NaN and enormous alike — after the cast there is nothing left to
    # test, because the garbage index is indistinguishable from a real one.
    ok = (
        np.isfinite(sx) & np.isfinite(sy)
        & (sx >= 0.0) & (sx < width)
        & (sy >= 0.0) & (sy < height)
    )
    sx = np.where(ok, sx, 0.0)
    sy = np.where(ok, sy, 0.0)

    ix = sx.astype(np.int32)
    iy = sy.astype(np.int32)
    visible = ok & in_front

    ix = ix[visible]
    iy = iy[visible]
    # `scale` is the perspective falloff. Additive accumulation means no depth
    # sort is needed — light from near and far stars simply sums, which is what
    # light does. A painter's-algorithm sort would cost an argsort per frame and
    # look worse.
    light = colour[visible] * (brightness[visible] * scale[visible])[:, None] * C.EXPOSURE

    # Scatter-add. A plain `buffer[iy, ix] += light` silently drops every
    # overlap — repeated indices overwrite rather than accumulate — and overlap
    # is precisely what draws the arms and the bulge.
    #
    # np.add.at does it correctly but runs unbuffered and is very slow.
    # np.bincount over flattened indices is the same operation an order of
    # magnitude faster, at the cost of doing it per channel.
    # The buffer is (width, height, 3) to match surfarray's own layout. Storing
    # it row-major and transposing at the end looks tidier and costs a full
    # non-contiguous copy of 2.9M floats every frame.
    flat = ix * height + iy

    # Accumulate over *occupied* pixels only. Stars cover about 3.6% of the
    # canvas, so a bincount with minlength=960000 spends nearly all its time
    # allocating and zeroing a 7.7 MB float64 array — three times, every frame,
    # at a cost independent of how many stars there are. Collapsing to the
    # unique indices first makes the work proportional to the galaxy instead of
    # to the window.
    #
    # Indices are unique after this, so plain assignment is correct; the
    # accumulation of overlapping stars has already happened in the bincount.
    buffer.fill(0.0)
    uniq, inv = np.unique(flat, return_inverse=True)
    summed = np.empty((uniq.size, 3), dtype=np.float32)
    for ch in range(3):
        summed[:, ch] = np.bincount(inv, weights=light[:, ch], minlength=uniq.size)
    buffer.reshape(-1, 3)[uniq] = summed

    # Scale and clamp in one pass over the buffer rather than two.
    np.multiply(buffer, 255.0, out=buffer)
    np.clip(buffer, 0.0, 255.0, out=buffer)
    pygame.surfarray.blit_array(surface, buffer.astype(np.uint8))
    return int(visible[:star_count].sum()), buffer


def new_buffer():
    # (width, height, 3) — surfarray's layout, so no transpose before blitting.
    return np.zeros((C.WINDOW_WIDTH, C.WINDOW_HEIGHT, 3), dtype=np.float32)
