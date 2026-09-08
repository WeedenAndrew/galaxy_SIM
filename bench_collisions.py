"""Gate 2: correctness checks and settled-galaxy scaling (no rendering)."""
import time
import numpy as np
import config as C
import collisions
import galaxy
import gravity
import physics


def check():
    rng = np.random.default_rng(41)
    pos = rng.uniform(-2, 2, (200, 3))
    cell = np.floor(pos).astype(np.int64)
    expected = {(i,j) for i in range(len(pos)) for j in range(i+1,len(pos))
                if np.array_equal(cell[i], cell[j]) and np.linalg.norm(pos[i]-pos[j]) < 1}
    assert set(map(tuple, collisions.detect(pos, 1))) == expected
    bad = np.array([[np.nan,0,0],[np.inf,0,0],[-np.inf,0,0],[1e300,0,0],[-1e300,0,0]])
    pair = np.array([[0.1,0.1,0.1],[0.2,0.2,0.2]])
    combined = np.concatenate((pair,bad))
    before = combined.copy()
    assert collisions.detect(combined,1).tolist() == [[0,1]]
    assert np.array_equal(combined,before,equal_nan=True)
    assert len(collisions.detect(np.array([[.99,0,0],[1.01,0,0]]),1)) == 0
    assert len(collisions.detect(np.empty((0,3)),1)) == 0
    assert len(collisions.detect(np.zeros((10,3)),1)) == 45
    edge = np.array([[-2**20,0,0],[-2**20+.1,0,0],[2**20,0,0]])
    assert collisions.detect(edge,1).tolist() == [[0,1]]
    for radius in (0,-1,np.nan,np.inf):
        try: collisions.detect(pair,radius)
        except ValueError: pass
        else: raise AssertionError('invalid radius accepted')
    assert collisions.detect(bad,1e-300).shape == (0,2)
    print('PASS: brute-force same-cell oracle, boundary omission, NaN/inf/huge, key limits, empty/dense, input unchanged')


def run():
    check()
    original_count = C.STAR_COUNT
    try:
        for seed in (7,19):
            for n in (10000,20000,25000,40000):
                C.STAR_COUNT = n  # benchmark only; config file never changes
                rng=np.random.default_rng(seed)
                pos,vel,_,_=galaxy.build(rng)
                pm=gravity.PMGravity()
                mass=C.STELLAR_MASS/n
                galaxy.rebalance_velocities(pos,vel,pm,mass,rng)
                acc=physics.acceleration(pos,0,pm,mass)
                t=0.; dt=C.SIM_SPEED/C.TARGET_FPS/C.SUBSTEPS
                for _ in range(300):
                    for _ in range(C.SUBSTEPS):
                        t,acc=physics.step(pos,vel,dt,t,acc,pm,mass)
                totals=np.zeros(3); event_r=[]
                for _ in range(100):
                    for _ in range(C.SUBSTEPS):
                        t,acc=physics.step(pos,vel,dt,t,acc,pm,mass)
                    start=time.perf_counter()
                    pairs,k=collisions.detect(pos,return_candidates=True)
                    elapsed=time.perf_counter()-start
                    totals += (len(pairs),elapsed*1000,k)
                    if len(pairs):
                        event_r.extend(np.linalg.norm((pos[pairs[:,0]]+pos[pairs[:,1]])/2,axis=1)/C.MPP)
                    assert np.isfinite(pos).all()
                e,ms,k=totals/100
                print(f'seed={seed} N={n} events={e:.2f} detect_ms={ms:.3f} candidates={k:.2f} '
                      f'event_median_r={np.median(event_r):.2f}px star_median_r={np.median(np.linalg.norm(pos,axis=1))/C.MPP:.2f}px finite=True',flush=True)
    finally:
        C.STAR_COUNT=original_count


if __name__ == '__main__':
    run()
