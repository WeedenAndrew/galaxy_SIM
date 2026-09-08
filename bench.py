r"""Headless benchmark. Run this before believing any frame rate.

    $env:SDL_VIDEODRIVER='dummy'; .\.venv\Scripts\python.exe -B bench.py

Note the r-prefix on this docstring. Without it Python reads the `\v` in
`.venv` as a vertical tab and warns about `\.` and `\S` as invalid escapes —
which becomes an error in a future version. Any Windows path in a Python string
wants a raw string or forward slashes.

Frame rate is a property of the machine. Figures measured in a Linux container
during development ran about 60% faster than the target Windows box, which was
misleading for long enough to matter — so the only numbers worth acting on are
the ones this prints on the machine you will actually run on.

`-B` keeps it from writing bytecode next to the source.
"""

import argparse
import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame

import config as C
import galaxy
import gravity
import physics
import render
from camera import Camera
from flashes import CollisionEffects


def bench(warmup=60, frames=300, seed=7, collisions_enabled=None):
    if collisions_enabled is None:
        collisions_enabled = C.COLLISIONS_ENABLED
    pygame.init()
    surface = pygame.display.set_mode((C.WINDOW_WIDTH, C.WINDOW_HEIGHT))

    rng = np.random.default_rng(seed)
    pos, vel, colour, brightness = galaxy.build(rng)
    star_mass = C.STELLAR_MASS / C.STAR_COUNT
    pm = gravity.PMGravity()
    galaxy.rebalance_velocities(pos, vel, pm, star_mass, rng)
    acc = physics.acceleration(pos, 0.0, pm, star_mass)
    cam = Camera()
    buffer = render.new_buffer()
    effects = CollisionEffects()
    print(f"collisions {'ON' if collisions_enabled else 'OFF'}")

    print(f"backend {gravity.FFT_BACKEND}, workers {C.FFT_WORKERS}")
    print(
        f"STAR_COUNT={C.STAR_COUNT:,}  PM_GRID={C.PM_GRID}  "
        f"cell={pm.cell / C.MPP:.1f}px  SUBSTEPS={C.SUBSTEPS}  "
        f"surface={surface.get_size()}"
    )

    # Warm up: first calls build FFT plans and touch pages for the first time,
    # and including them inflates the mean by a wide margin.
    t = 0.0
    sub_dt = (1.0 / 60.0) * C.SIM_SPEED / C.SUBSTEPS
    for _ in range(warmup):
        t, acc = physics.step(pos, vel, sub_dt, t, acc, pm, star_mass)
        if collisions_enabled:
            effects.advance(pos, 1.0 / 60.0)
        _, buffer = render.draw(surface, pos, colour, brightness, buffer, cam,
                                effects.pool if collisions_enabled else None)

    t_step = t_draw = 0.0
    for _ in range(frames):
        t0 = time.perf_counter()
        t, acc = physics.step(pos, vel, sub_dt, t, acc, pm, star_mass)
        t1 = time.perf_counter()
        if collisions_enabled:
            effects.advance(pos, 1.0 / 60.0)
        drawn, buffer = render.draw(surface, pos, colour, brightness, buffer, cam,
                                    effects.pool if collisions_enabled else None)
        t2 = time.perf_counter()
        t_step += t1 - t0
        t_draw += t2 - t1

    step_ms = t_step / frames * 1000.0
    draw_ms = t_draw / frames * 1000.0
    total = step_ms + draw_ms

    print(f"\n  physics step   {step_ms:6.2f} ms")
    print(f"  render/effects {draw_ms:6.2f} ms")
    print(f"  total          {total:6.2f} ms   ->  {1000.0 / total:5.1f} fps")
    print(f"  budget         {16.67 - total:+6.2f} ms against 60 fps")

    r = np.hypot(pos[:, 0], pos[:, 1])
    print(
        f"\n  finite={np.isfinite(pos).all()}  drawn={drawn:,}  "
        f"escaped={pm.escaped}  mean radius={r.mean() / C.MPP:.1f}px  "
        f"|z|={np.abs(pos[:, 2]).mean() / C.MPP:.2f}px"
    )
    # Excludes HUD text, event handling and display flip — those are small, but
    # this is a floor rather than the full frame.
    pygame.quit()
    return total


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--warmup", type=int, default=60)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--collisions", choices=("on", "off"), default=None)
    a = p.parse_args()
    bench(a.warmup, a.frames, a.seed,
          None if a.collisions is None else a.collisions == "on")
