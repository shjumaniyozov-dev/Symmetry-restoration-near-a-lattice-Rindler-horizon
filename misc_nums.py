import numpy as np
from qp import *
d=np.load('rind_L192.npz'); ts=d['ts']; y=d['res'][0,3]
m=(ts>=1.2-1e-9)&(ts<=2.8+1e-9); print('lattice slope l=192:',-np.polyfit(ts[m],y[m],1)[0])
tq=np.linspace(1.2,2.8,9); q=[dS2_qp(W_rind(K,t,0,192),np.pi/2) for t in tq]; print('QP slope:',-np.polyfit(tq,q,1)[0])
def crossings(ts,a,b):
    d=b-a; out=[]
    for i in range(1,len(ts)):
        if np.sign(d[i-1])!=np.sign(d[i]) and d[i-1]!=0: out.append(round(ts[i-1]+(ts[i]-ts[i-1])*d[i-1]/(d[i-1]-d[i]),2))
    return out
for f,gi,tmax in [('rind_N6000.npz',1,4.0),('rind_L96.npz',1,3.4),('rind_L192.npz',1,3.6)]:
    dd=np.load(f); ts=dd['ts']; mm=ts<=tmax+1e-9
    print(f,dd['geoms'][gi],'crossings (0.2,0.3):',crossings(ts[mm],dd['res'][gi,0,mm],dd['res'][gi,1,mm]))
# where does lattice fall below QP for rho=1
for f,tmax,l in [('rind_N6000.npz',4.0,48),('rind_L96.npz',3.6,96),('rind_L192.npz',3.6,192)]:
    dd=np.load(f); ts=dd['ts']
    for i in range(4):
        q=np.array([dS2_qp(W_rind(K,t,0,l),dd['ths'][i]) for t in ts]); dif=dd['res'][0,i]-q
        below=[t for t,v in zip(ts,dif) if v<0 and t<=tmax]
        print(l,round(dd['ths'][i]/np.pi,1),'first t with lattice<QP:',below[:1])
