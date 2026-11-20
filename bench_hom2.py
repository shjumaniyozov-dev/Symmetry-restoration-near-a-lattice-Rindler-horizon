import numpy as np
from gauss import *; from qp import *
th=0.3*np.pi; c=np.cos(th)
for l in [40,80,160]:
    N=2*l+2*int(1.2*l)+40
    G0=ground_gamma(xy_majorana_M(N,np.sin(th)**2/(1+c**2),2*c/(1+c**2)),ref=gamma_tilted(N,th))
    ev=Evolver(hop_matrix(np.ones(N-1)))
    A=range(N//2-l//2,N//2+l//2); idx=np.ravel([[2*j,2*j+1] for j in A])
    out=[]
    for z in [0.25,0.5,0.75,1.0]:
        t=z*l; R=ev.R(t)[idx]; GA=R@G0@R.T
        out.append((dS2(GA),dS2_qp(W_hom(K,t,l),th)))
    print('l=',l,' '.join(f"z={z}: ex={a:.4f} qp={b:.4f} d={a-b:+.4f}" for z,(a,b) in zip([0.25,0.5,0.75,1.0],out)))
