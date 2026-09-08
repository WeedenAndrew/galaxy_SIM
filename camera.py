"""Orbit camera. Yaw and pitch around the galactic centre, plus zoom.

Replaces the fixed TILT projection, which could only ever show the galaxy from
one angle. The disc is a genuinely three-dimensional object — thin, warped at
the edges, with a rounder bulge — and none of that reads face-on.

Perspective rather than orthographic: with a flat projection a tilted disc is
just an ellipse, and there is no depth cue at all. Perspective makes the near
side visibly larger, which is what tells you which way the thing is facing.
"""

import math
import numpy as np
import config as C


class Camera:
    def __init__(self):
        self.yaw = 0.0
        self.pitch = C.TILT
        self.distance = C.CAM_START_DISTANCE
        self._dirty = True
        self._basis = None

    # ── controls ────────────────────────────────────────────────────────────
    def orbit(self, dyaw, dpitch):
        self.yaw += dyaw
        # Stop just short of the poles. At exactly +/-90 degrees the up-vector
        # becomes parallel to the view direction and the basis degenerates.
        limit = math.pi / 2 - 1e-3
        self.pitch = max(-limit, min(limit, self.pitch + dpitch))
        self._dirty = True

    def zoom(self, factor):
        self.distance = float(
            np.clip(self.distance * factor, C.CAM_MIN_DISTANCE, C.CAM_MAX_DISTANCE)
        )
        self._dirty = True

    # ── projection ──────────────────────────────────────────────────────────
    def _rebuild(self):
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        # Camera axes in world space.
        # `offset` points from the origin out to the eye. The camera then looks
        # back down it, so the view axis is -offset, not +offset. Using the
        # same vector for both puts the entire galaxy behind the near plane and
        # renders nothing at all — with no error, because "zero stars visible"
        # is a perfectly valid frame.
        offset = np.array([cp * cy, cp * sy, sp])
        view = -offset
        right = np.array([-sy, cy, 0.0])
        up = np.cross(view, right)
        self._basis = np.stack([right, up, view]).astype(np.float64)
        self._eye = offset * self.distance
        self._dirty = False

    def project(self, pos, width=None, height=None):
        """World (N,3) metres -> screen x, y, depth, and a per-star size scale.

        Takes the viewport size rather than reading it from config, because the
        window is resizable and the real surface is the only authority on how
        big it is.

        Returns `scale` so the renderer can brighten near stars and dim far
        ones. Without it a perspective view looks like a flat sprite sheet.
        """
        width = C.WINDOW_WIDTH if width is None else width
        height = C.WINDOW_HEIGHT if height is None else height
        if self._dirty:
            self._rebuild()

        # Sanitise before the matmul, not after. A single NaN coordinate
        # contaminates that star's whole row, and the warning surfaces as
        # "invalid value encountered in matmul" — which names the projection
        # rather than whatever actually produced the NaN. Substituting zero
        # would place the star at the origin, which is *visible*, so the
        # finiteness mask has to be carried through to the visibility test.
        finite = np.all(np.isfinite(pos), axis=1)
        rel = np.where(finite[:, None], pos - self._eye, 0.0)
        cam = rel @ self._basis.T          # (N,3): right, up, forward
        depth = cam[:, 2]

        # Behind the camera, or so close the divide explodes.
        near = C.CAM_NEAR * C.MPP
        safe = np.maximum(depth, near)

        f = C.CAM_FOCAL * height
        sx = cam[:, 0] / safe * f + width * 0.5
        sy = -cam[:, 1] / safe * f + height * 0.5

        # Inverse-square falloff, normalised so a star at the orbit distance
        # keeps roughly unit brightness however far you zoom.
        scale = (self.distance / safe) ** 2
        visible = (depth > near) & finite
        return sx, sy, depth, scale, visible
