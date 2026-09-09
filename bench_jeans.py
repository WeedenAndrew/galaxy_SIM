"""Gate 2: setup timings, Jeans quadrature convergence and bulge moments."""
import time
import numpy as np
import config as C
import galaxy
import gravity
import physics
import jeans


def run():
    queries=np.array([5,15,45,100,200,400])*C.MPP
    print('sigma km/s at 5,15,45,100,200,400 px; 512 radii, 64 directions, outer=500 px')
    original_assign=jeans.assign
    capture={}
    def checked_assign(pos,vel,pm,mass,rng):
        disc=vel[:len(pos)-int(len(pos)*.18)].copy()
        start=time.perf_counter()
        original_assign(pos,vel,pm,mass,rng)
        capture['ms']=(time.perf_counter()-start)*1000
        assert np.array_equal(disc,vel[:len(disc)])
    jeans.assign=checked_assign
    try:
        for seed in (7,19,41,73,101):
            rng=np.random.default_rng(seed); pos,vel,_,_=galaxy.build(rng)
            pm=gravity.PMGravity(); mass=C.STELLAR_MASS/C.STAR_COUNT
            galaxy.rebalance_velocities(pos,vel,pm,mass,rng)
            r,s,g=jeans.profile(pos,pm,mass)
            count=int(len(pos)*.18)
            scaled=vel[-count:]/np.interp(np.linalg.norm(pos[-count:],axis=1),r,s)[:,None]
            assert np.isfinite(vel).all()
            assert np.max(np.abs(scaled.mean(axis=0)))<.06
            assert np.max(np.abs(scaled.std(axis=0)-1))<.06
            print(f'seed={seed} sigma={np.round(np.interp(queries,r,s)/1000,3)} '
                  f'added_setup={capture["ms"]:.3f}ms disc_identical=True finite=True '
                  f'normalized_component_std={np.round(scaled.std(axis=0),4)}',flush=True)
            if seed==7:
                reference=(pos,pm,mass,r,s)
    finally:
        jeans.assign=original_assign
    pos,pm,mass,r,s=reference
    base=np.interp(queries,r,s)
    for kwargs in ({'directions':128},{'directions':256}, {'radial_points':1024}):
        start=time.perf_counter(); rr,ss,_=jeans.profile(pos,pm,mass,**kwargs)
        values=np.interp(queries,rr,ss)
        print(f'convergence {kwargs}: sigma={np.round(values/1000,3)} '
              f'max_change={100*np.max(np.abs(values/base-1)):.4f}% time={(time.perf_counter()-start)*1000:.3f}ms',flush=True)
    source_force=pm.accelerate(pos,mass).copy()
    query_force=pm.accelerate_at(pos,mass,pos)
    assert np.array_equal(source_force,query_force)
    small=pos[:100].copy(); original=small.copy()
    q1=pm.accelerate_at(pos,mass,small)
    q2=pm.accelerate_at(pos,mass,np.concatenate((small,small)))
    assert np.array_equal(q1,q2[:100]) and np.array_equal(q1,q2[100:])
    assert np.array_equal(small,original)
    assert np.array_equal(source_force,pm.accelerate(pos,mass))
    print('PASS source/probe equality, duplicated massless probes unchanged, inputs unchanged, no force-cache corruption')


if __name__=='__main__':
    run()
