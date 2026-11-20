import numpy as np
from functools import reduce
from gauss import *
sx=np.array([[0,1],[1,0]],complex); sy=np.array([[0,-1j],[1j,0]]); sz=np.diag([1.,-1.]).astype(complex); I2=np.eye(2)
def op(o,j,N): return reduce(np.kron,[o if l==j else I2 for l in range(N)])
import sys
N=int(sys.argv[1]); th=0.3*np.pi; c=np.cos(th)
# JW Majoranas: w_{2j-1}=S_j sx_j, w_{2j}=S_j sy_j
def S(j): return reduce(lambda a,b:a@b,[op(sz,l,N) for l in range(j)],np.eye(2**N))
W=[]
for j in range(N):
    W.append(S(j)@op(sx,j,N)); W.append(S(j)@op(sy,j,N))
site=np.array([np.cos(th/2),np.sin(th/2)],complex); sitem=np.array([np.cos(th/2),-np.sin(th/2)],complex)
p=reduce(np.kron,[site]*N); m=reduce(np.kron,[sitem]*N)
cat=(p+m); cat/=np.linalg.norm(cat)
def Gam(psi):
    G=np.zeros((2*N,2*N))
    for a in range(2*N):
        for b in range(2*N):
            if a!=b: G[a,b]=(-1j*np.vdot(psi,W[a]@W[b]@psi)).real
    return G
Gc=Gam(cat); Ga=gamma_tilted(N,th)
print('max|Gamma_cat(ED)-Gamma_analytic| =',np.abs(Gc-Ga).max(), ' (expected O(cos^N)=%.1e)'%c**N)
print('purity check eig(iG):',np.sort(np.abs(np.linalg.eigvalsh(1j*Gc)))[:3])
# time evolution with Rindler hopping, compare ED vs Gaussian
J=np.arange(1,N)*1.0
Hs=np.zeros((2**N,2**N),complex)
for j in range(N-1):   # -1/2 J (c^dag c + h.c.) = -J/4 (sx sx + sy sy)
    Hs+= J[j]/4*(op(sx,j,N)@op(sx,j+1,N)+op(sy,j,N)@op(sy,j+1,N))
ev=Evolver(hop_matrix(J))
t=0.7
w,v=np.linalg.eigh(Hs); Ut=(v*np.exp(-1j*w*t))@v.conj().T
ct=Ut@cat
print('max|Gamma(t) ED - Gaussian| =',np.abs(Gam(ct)-ev.evolve(Gc,t)).max())
# asymmetry of A = first 3 sites: ED (spin reduced density matrix = fermionic for left block)
lA=3
def rhoA(psi):
    M=psi.reshape(2**lA,2**(N-lA)); return M@M.conj().T
Q=sum(op((sz+I2)/2,j,lA) for j in range(lA)).real.diagonal()
def dS2_ED(r):
    rq=np.zeros_like(r); 
    for q in np.unique(Q):
        P=np.diag((Q==q).astype(float)); rq+=P@r@P
    return np.log(np.trace(r@r).real)-np.log(np.trace(rq@rq).real)
for psi,name in [(ct,'cat(t)'),(Ut@p,'product(t)')]:
    print(name,'ED dS2 =',dS2_ED(rhoA(psi)))
print('Gaussian dS2 cat(t) =',dS2(restrict(ev.evolve(Gc,t),range(lA))))
np.set_printoptions(precision=3,suppress=True,linewidth=150)
print(Gc[:6,:6]); print(Ga[:6,:6])
print('evolve with -t:',np.abs(Gam(ct)-ev.evolve(Gc,-t)).max())
