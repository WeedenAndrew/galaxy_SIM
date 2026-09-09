"""Stage 1: truncated radial CDF and raw mesh/isolated Plummer comparison."""
import time
import numpy as np
import config as C
import galaxy
import gravity


def run():
    def cdf(r):
        return (r / np.sqrt(r*r+C.BULGE_R*C.BULGE_R))**3
    umax=cdf(C.MAX_R)
    nbulge=int(C.STAR_COUNT*.18)
    print(f'N={C.STAR_COUNT} bulge={nbulge} grid={C.PM_GRID} umax={umax:.9f}')
    all_r=[]; all_phi=[]; costs=[]
    for seed in (7,19,41,73,101):
        p,_,_,_=galaxy.build(np.random.default_rng(seed))
        p=p[-nbulge:].copy(); r=np.linalg.norm(p,axis=1)
        # Recover the original inverse-CDF draw from the new radius. The old
        # clipped version would set every draw above F(MAX_R) to MAX_R.
        u=cdf(r)/umax
        old_r=C.BULGE_R*np.sqrt(u**(2/3)/(1-u**(2/3)))
        old_r=np.minimum(old_r,C.MAX_R)
        print(f'seed={seed} within_1px_before={np.count_nonzero(old_r>=C.MAX_R-C.MPP)} '
              f'after={np.count_nonzero(r>=C.MAX_R-C.MPP)} '
              f'shell_after={np.count_nonzero(np.isclose(r,C.MAX_R,rtol=1e-12,atol=0))}')
        pm=gravity.PMGravity()
        start=time.perf_counter(); phi=pm.potential(p,C.BULGE_MASS/nbulge); costs.append((time.perf_counter()-start)*1000)
        assert np.isfinite(phi).all() and np.all(r<C.MAX_R)
        all_r.append(r); all_phi.append(phi)
        if seed==7:
            print('radius_px empirical_CDF isolated_CDF conditional_CDF')
            for radius in (15,25,45,100,200,400,499,500):
                f=cdf(radius*C.MPP)
                print(f'{radius:3} {np.mean(r<=radius*C.MPP):.6f} {f:.6f} {f/umax:.6f}')
    r=np.concatenate(all_r); phi=np.concatenate(all_phi)
    analytic=-C.G*C.BULGE_MASS/np.sqrt(r*r+C.BULGE_R*C.BULGE_R)
    edges=(0,25,46.6666666667,75,100,150,200,300,400,500)
    print('range_px count mean_radius_px mesh_phi analytic_phi ratio_of_means')
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(r>=lo*C.MPP)&(r<hi*C.MPP)
        print(f'{lo:5.1f}-{hi:5.1f} {mask.sum():5} {r[mask].mean()/C.MPP:8.3f} '
              f'{phi[mask].mean():.6e} {analytic[mask].mean():.6e} {phi[mask].mean()/analytic[mask].mean():.4f}')
    print('potential_ms (fresh solve, 4500 sources):', ', '.join(f'{x:.3f}' for x in costs))
    print('Comparison uses total deposited mass M=BULGE_MASS; analytic formula is the untruncated isolated Plummer. No fitted offset.')


if __name__ == '__main__':
    run()
