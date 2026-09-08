"""Gate 3 checks. Run with the project venv, -B -W error::RuntimeWarning."""
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
import time
import numpy as np
import pygame
import config as C
import flashes
import collisions
import bench_collisions
import galaxy
import gravity
import physics
import render
from camera import Camera


def check_pool():
    bench_collisions.check()
    p = flashes.FlashPool()
    pos = np.array([[2.,4.,6.],[4.,6.,8.]])
    pair = np.array([[0,1]])
    p.spawn(pos,pair)
    assert np.array_equal(p.pos[0], [3,5,7])
    p.update(C.FLASH_RISE)
    assert np.isclose(p.brightness[0],C.FLASH_PEAK)
    p.update(C.FLASH_LIFETIME)
    assert not np.any(p.brightness)
    p.clear()
    for i in range(C.MAX_FLASHES+3):
        pos.fill(i)
        p.spawn(pos,pair)
    assert p.pos[0,0] == C.MAX_FLASHES and p.pos[2,0] == C.MAX_FLASHES+2
    assert p.pos[3,0] == 3

    # Tripwire on every NumPy call used by update/spawn: output-producing calls
    # must use existing arrays. Also forbid explicit array constructors.
    arrays = {name:value for name,value in vars(p).items() if isinstance(value,np.ndarray)}
    original = {}
    calls = [0]
    def guard(fn):
        def checked(*args, **kwargs):
            out = kwargs.get('out')
            assert any(out is a for a in arrays.values()), 'missing preallocated out'
            result = fn(*args, **kwargs)
            assert result is out
            calls[0] += 1
            return result
        return checked
    def forbidden(*args, **kwargs):
        raise AssertionError('array construction in pool update/spawn')
    for name in ('add','minimum','divide','subtract','clip','multiply'):
        original[name] = getattr(np,name)
        setattr(np,name,guard(original[name]))
    for name in ('array','asarray','empty','zeros','ones','full','empty_like','zeros_like','concatenate','stack'):
        original[name] = getattr(np,name)
        setattr(np,name,forbidden)
    try:
        for frame in range(600):
            p.update(1/60)
            if frame % 4 == 0: p.spawn(pos,pair)
    finally:
        for name,fn in original.items(): setattr(np,name,fn)
    assert all(getattr(p,name) is a for name,a in arrays.items())
    print(f'PASS 600 frames: {calls[0]} array-producing calls reused explicit out; constructors blocked; pool arrays unchanged')
    print('PASS rise/expiry, midpoint, capacity overflow and oldest-first age ties')

    # Scheduler does not replay a batch, or detect while paused.
    original_detect = collisions.detect
    detections = [0]
    def fake(pos):
        detections[0] += 1
        return pair
    effects = flashes.CollisionEffects()
    collisions.detect = fake
    try:
        for _ in range(8): effects.advance(pos,1/60)
        assert detections[0] == 2 and effects.pool._next == 2
        for _ in range(60): effects.advance(pos,1/60,sample=False)
        assert detections[0] == 2 and not np.any(effects.pool.brightness)
    finally: collisions.detect = original_detect
    print('PASS interval=4, no batch replay, wall-clock expiry during pause')


def check_render():
    pygame.init()
    surface = pygame.display.set_mode((800,600))
    pool = flashes.FlashPool()
    pos = np.zeros((2,3)); colour=np.zeros((2,3)); brightness=np.zeros(2)
    pool.spawn(pos,np.array([[0,1]])); pool.update(C.FLASH_RISE)
    original_blit=pygame.surfarray.blit_array; original_count=np.bincount
    counts=[0,0]
    def blit(*args):
        counts[0]+=1
        return original_blit(*args)
    def count(*args,**kwargs):
        counts[1]+=1
        return original_count(*args,**kwargs)
    pygame.surfarray.blit_array=blit; np.bincount=count
    try:
        drawn,buf=render.draw(surface,pos,colour,brightness,render.new_buffer(),Camera(),pool)
    finally:
        pygame.surfarray.blit_array=original_blit; np.bincount=original_count
    assert counts == [1,3] and drawn == 2 and buf.shape == (800,600,3)
    assert buf[:,:,0].max() > 0 and buf[:,:,1].max() == 0 and buf[:,:,2].max() > 0
    print('PASS render: one blit, three shared bincount calls, magenta flash, resize, star count excludes flashes')
    pygame.quit()


def benchmark():
    for seed in (7,19):
        rng=np.random.default_rng(seed)
        pos,vel,_,_=galaxy.build(rng); pm=gravity.PMGravity(); mass=C.STELLAR_MASS/C.STAR_COUNT
        galaxy.rebalance_velocities(pos,vel,pm,mass,rng)
        acc=physics.acceleration(pos,0,pm,mass); t=0; dt=C.SIM_SPEED/C.TARGET_FPS
        for _ in range(300): t,acc=physics.step(pos,vel,dt,t,acc,pm,mass)
        pool=flashes.FlashPool(); update=[]; detection=[]; spawn=[]; events=0
        for frame in range(600):
            t,acc=physics.step(pos,vel,dt,t,acc,pm,mass)
            start=time.perf_counter(); pool.update(1/60); update.append((time.perf_counter()-start)*1000)
            if frame % C.COLLISION_INTERVAL == 0:
                start=time.perf_counter(); pairs=collisions.detect(pos); detection.append((time.perf_counter()-start)*1000)
                start=time.perf_counter(); pool.spawn(pos,pairs); spawn.append((time.perf_counter()-start)*1000)
                events+=len(pairs)
        print(f'seed={seed} pool={C.MAX_FLASHES} frames=600 passes={len(detection)} events={events} '
              f'update_mean={np.mean(update):.4f}ms spawn_mean/pass={np.mean(spawn):.4f}ms '
              f'detect_mean/pass={np.mean(detection):.3f}ms p95={np.percentile(detection,95):.3f}ms max={max(detection):.3f}ms '
              f'detect_amortized={sum(detection)/600:.3f}ms finite={np.isfinite(pos).all()}',flush=True)


if __name__ == '__main__':
    check_pool()
    check_render()
    benchmark()
