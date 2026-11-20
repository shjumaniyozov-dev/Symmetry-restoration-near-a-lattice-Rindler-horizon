# brute-force counting of pairs (direct ODE integration of Hamilton eqs) vs closed form W
import numpy as np
from scipy.integrate import solve_ivp
g=1.0; xa,xb=20.,60.
def rhs(t,y): x,k=y; return [g*x*np.sin(k), g*np.cos(k)]
for k0 in [0.3,1.0,1.4,2.5]:
    for T in [0.5,1.5,3.0]:
        x0s=np.linspace(1e-3,xb,40001); dx=x0s[1]-x0s[0]
        # trajectories scale linearly in x0 (equations homogeneous in x): integrate once with x0=1
        xp=solve_ivp(rhs,[0,T],[1.0,k0],rtol=1e-11,atol=1e-12).y[0,-1]
        xm=solve_ivp(rhs,[0,T],[1.0,-k0],rtol=1e-11,atol=1e-12).y[0,-1]
        inside=(x0s*xp<xb)&(x0s*xp>xa)&(x0s*xm<xb)&(x0s*xm>xa)
        Wnum=inside.sum()*dx
        l=xb-xa; xbar=(xa+xb)/2
        Wcf=max(l-2*xbar*np.tanh(g*T)*abs(np.sin(k0)),0)*np.cosh(g*T)/(np.cosh(g*T)**2*np.cos(k0)**2+np.sin(k0)**2)
        print(f"k0={k0} t={T}: W_count={Wnum:.4f}  W_closed={Wcf:.4f}")
