"""Galaxy Sim — entry point.

    python main.py

Mouse
    drag        orbit the camera
    scroll      zoom
Keys
    SPACE       pause
    G           reverse stellar mesh gravity (nonphysical demonstration)
    H           toggle decorative collision flashes (off by default)
    A           toggle live mesh vs. analytic rails
    R           rebuild
    F           reset the view
    ESC         quit
"""

import os
import sys

import numpy as np
import pygame

import config as C
import galaxy
import gravity
import physics
import render
from flashes import CollisionEffects
from camera import Camera


def main():
    pygame.init()
    screen = pygame.display.set_mode(
        (C.WINDOW_WIDTH, C.WINDOW_HEIGHT), pygame.RESIZABLE
    )
    pygame.display.set_caption("Galaxy Sim")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 22)

    rng = np.random.default_rng()
    pos, vel, colour, brightness = galaxy.build(rng)
    star_mass = C.STELLAR_MASS / C.STAR_COUNT

    pm = gravity.PMGravity()
    live = C.USE_LIVE_GRAVITY
    polarity = 1.0
    if live:
        galaxy.rebalance_velocities(pos, vel, pm, star_mass, rng)
    acc = physics.acceleration(pos, 0.0, pm if live else None, star_mass, polarity)

    cam = Camera()
    buffer = render.new_buffer()
    effects = CollisionEffects()
    collisions_enabled = C.COLLISIONS_ENABLED

    sim_time = 0.0
    paused = False
    dragging = False
    running = True

    while running:
        dt = clock.tick(C.TARGET_FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    dragging = True
                elif event.button == 4:
                    cam.zoom(1.0 / C.CAM_ZOOM_STEP)
                elif event.button == 5:
                    cam.zoom(C.CAM_ZOOM_STEP)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                dx, dy = event.rel
                cam.orbit(-dx * C.CAM_ORBIT_SPEED, dy * C.CAM_ORBIT_SPEED)
            elif event.type == pygame.MOUSEWHEEL:
                cam.zoom(C.CAM_ZOOM_STEP ** -event.y)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_g:
                    polarity = -polarity
                    acc = physics.acceleration(
                        pos, sim_time, pm if live else None, star_mass, polarity
                    )
                elif event.key == pygame.K_h:
                    collisions_enabled = not collisions_enabled
                    effects.clear()
                elif event.key == pygame.K_a:
                    live = not live
                    acc = physics.acceleration(
                        pos, sim_time, pm if live else None, star_mass, polarity
                    )
                elif event.key == pygame.K_f:
                    cam = Camera()
                elif event.key == pygame.K_r:
                    pos, vel, colour, brightness = galaxy.build(rng)
                    sim_time = 0.0
                    effects.clear()
                    if live:
                        galaxy.rebalance_velocities(pos, vel, pm, star_mass, rng)
                    acc = physics.acceleration(
                        pos, 0.0, pm if live else None, star_mass, polarity
                    )

        if not paused:
            # dt is clamped. Dragging the window puts Windows into a modal
            # message loop and SDL gets no cycles until you let go — every
            # pygame app freezes there. What the clamp prevents is the *lurch*
            # afterwards: without it the integrator receives one enormous step
            # and flings the disc apart. Simulated time runs slightly slow
            # instead, which is the better failure.
            sub_dt = min(dt, 1.0 / 30.0) * C.SIM_SPEED / C.SUBSTEPS
            for _ in range(C.SUBSTEPS):
                sim_time, acc = physics.step(
                    pos, vel, sub_dt, sim_time, acc,
                    pm if live else None, star_mass, polarity,
                )

        # Wall-clock decay continues while paused; paused particles are not resampled.
        if collisions_enabled:
            effects.advance(pos, dt, sample=not paused)
        drawn, buffer = render.draw(
            screen, pos, colour, brightness, buffer, cam,
            effects.pool if collisions_enabled else None,
        )

        kpc = cam.distance / (1000 * C.PARSEC)
        hud = (
            f"{clock.get_fps():5.1f} fps   {C.STAR_COUNT:,} stars   "
            f"{drawn:,} visible   {sim_time / (1e6 * C.YEAR):,.0f} Myr   "
            f"{kpc:,.0f} kpc out   "
            f"{pm.escaped} escaped   "
            f"gravity {'LIVE' if live else 'analytic'}"
            f" mesh {'attract' if polarity > 0 else 'repel (nonphysical)'}"
            f" flashes {'ON' if collisions_enabled else 'OFF'}"
            f"{'   [PAUSED]' if paused else ''}"
        )
        screen.blit(font.render(hud, True, (170, 180, 200)), (14, 12))
        screen.blit(
            font.render("drag orbit · scroll zoom · G polarity · H flashes · A rails · F reset",
                        True, (90, 100, 120)),
            (14, screen.get_height() - 26),
        )
        pygame.display.flip()

    pygame.quit()


def _run():
    """Entry point with crash logging.

    A windowed build has no console, so an unhandled exception closes the
    window with nothing to look at — the failure is invisible rather than
    absent. Write the traceback beside the executable instead, where it can be
    read after the fact.
    """
    try:
        main()
    except Exception:
        import datetime
        import traceback

        # sys.frozen is set by PyInstaller; sys.executable is then the .exe
        # itself rather than a Python interpreter, so the log lands next to
        # what the user actually double-clicked.
        base = (
            os.path.dirname(sys.executable)
            if getattr(sys, "frozen", False)
            else os.path.dirname(os.path.abspath(__file__))
        )
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(base, f"galaxy-sim-crash-{stamp}.log")
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(traceback.format_exc())
        except OSError:
            pass  # read-only directory; the re-raise below still reports it
        raise


if __name__ == "__main__":
    _run()
