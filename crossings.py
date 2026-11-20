import numpy as np
from qp import *
from scipy.optimize import brentq
def first_cross(ts,a,b):
    d=b-a   # b: larger theta
    for i in range(1,len(ts)):
        if d[i-1]>0 and d[i]<=0:
            return ts[i-1]+(ts[i]-ts[i-1])*d[i-1]/(d[i-1]-d[i])
    return np.nan
sets=[('rind_N6000.npz',[(0,4.0),(1,4.0),(2,3.8)]),('rind_L96.npz',[(0,3.6),(1,3.4),(2,3.2)]),('rind_L192.npz',[(0,3.6),(1,3.6)])]
for f,gl in sets:
    d=np.load(f); ths=d['ths']; ts=d['ts']
    for gi,tmax in gl:
        xa,xb=d['geoms'][gi]; m=ts<=tmax+1e-9; rho=(xb-xa)/(xa+xb)
        tq=np.linspace(0,tmax,int(tmax/0.02)+1)
        Q=[np.array([dS2_qp(W_rind(K,t,xa,xb),th) for t in tq]) for th in ths]
        out=[]
        for i in range(4):
            for j in range(i+1,4):
                te=first_cross(ts[m],d['res'][gi,i,m],d['res'][gi,j,m]); tqv=first_cross(tq,Q[i],Q[j])
                out.append(f"({ths[i]/np.pi:.1f},{ths[j]/np.pi:.1f}): ex {te:.2f} qp {tqv:.2f}")
        print(f"A=[{xa},{xb}] l={xb-xa} rho={rho:.3f} | "+' ; '.join(out))
