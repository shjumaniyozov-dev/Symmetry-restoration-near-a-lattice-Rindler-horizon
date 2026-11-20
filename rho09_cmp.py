import numpy as np
from qp import *
from misc_nums import crossings
d=np.load('rind_rho09.npz'); ts=d['ts']; ths=d['ths']; R=d['res'][0]
tq=np.linspace(0,3.6,181)
Q=[np.array([dS2_qp(W_rind(K,t,10,190),th) for t in tq]) for th in ths]
for i in range(4):
    print(f"th={ths[i]/np.pi:.1f} ex:",' '.join(f'{v:.3f}' for v in R[i,::2])); print('        qp:',' '.join(f'{v:.3f}' for v in Q[i][::20]))
for i in range(4):
    for j in range(i+1,4):
        print((round(ths[i]/np.pi,1),round(ths[j]/np.pi,1)),'lat',crossings(ts,R[i],R[j]),'qp',crossings(tq,Q[i],Q[j]))
