import numpy as np, time, sys
from rind import *; from gauss import dS2
N=int(sys.argv[1]); tag=sys.argv[2]
ths=np.array([0.2,0.3,0.4,0.5])*np.pi
ts=np.round(np.arange(0,5.01,0.2),3)
geoms=[(0,48),(16,64),(48,96)]
t0=time.time(); m=Rindler(N); print('diag',time.time()-t0,flush=True)
res=np.zeros((len(geoms),len(ths),len(ts)))
for gi,(xa,xb) in enumerate(geoms):
    sites=list(range(xa,xb))        # 0-based site j <-> physical site j+1 ; xi in [xa,xb]
    for ti,t in enumerate(ts):
        R=m.RA(sites,t)
        for i,th in enumerate(ths):
            GA=apply_gamma_tilted(R,th)@R.T
            res[gi,i,ti]=dS2(GA)
        print(gi,t,res[gi,:,ti],round(time.time()-t0),flush=True)
np.savez(f'rind_{tag}.npz',res=res,ts=ts,ths=ths,geoms=np.array(geoms),N=N)
