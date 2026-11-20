import numpy as np
from functools import reduce
from gauss import *
sx=np.array([[0,1],[1,0]],complex); sy=np.array([[0,-1j],[1j,0]]); sz=np.diag([1.,-1.]).astype(complex); I2=np.eye(2)
def op(o,j,N): return reduce(np.kron,[o if l==j else I2 for l in range(N)])
N=6; th=0.3*np.pi
W=[]
for j in range(N):
    S=reduce(lambda a,b:a@b,[op(sz,l,N) for l in range(j)],np.eye(2**N))
    W+= [S@op(sx,j,N), S@op(sy,j,N)]
site=np.array([np.cos(th/2),np.sin(th/2)]); sitem=np.array([np.cos(th/2),-np.sin(th/2)])
for sign in [1,-1]:
    cat=reduce(np.kron,[site]*N)+sign*reduce(np.kron,[sitem]*N); cat=cat/np.linalg.norm(cat)
    Gc=np.array([[0 if a==b else (-1j*np.vdot(cat,W[a]@W[b]@cat)).real for b in range(2*N)] for a in range(2*N)])
    c=np.cos(th); G=ground_gamma(xy_majorana_M(N,np.sin(th)**2/(1+c**2),2*c/(1+c**2)),ref=Gc)
    print('cat sign',sign,' max|G_XYground - G_ED| =',np.abs(G-Gc).max())
