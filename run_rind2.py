import numpy as np, time, sys
from rind import *; from gauss import dS2
N=int(sys.argv[1]); tag=sys.argv[2]; geoms=eval(sys.argv[3]); tmax=float(sys.argv[4])
ths=np.array([0.2,0.3,0.4,0.5])*np.pi
ts=np.round(np.arange(0,tmax+1e-9,0.2),3)
t0=time.time(); m=Rindler(N); print('diag',time.time()-t0,flush=True)
res=np.zeros((len(geoms),len(ths),len(ts)))
for gi,(xa,xb) in enumerate(geoms):
    sites=list(range(xa,xb))
    for ti,t in enumerate(ts):
        R=m.RA(sites,t)
        for i,th in enumerate(ths):
            res[gi,i,ti]=dS2(apply_gamma_tilted(R,th)@R.T)
        print(gi,t,res[gi,:,ti],round(time.time()-t0),flush=True)
        np.savez(f'rind_{tag}.npz',res=res,ts=ts,ths=ths,geoms=np.array(geoms),N=N)
