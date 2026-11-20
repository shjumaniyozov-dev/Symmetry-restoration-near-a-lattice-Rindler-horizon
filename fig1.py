import numpy as np
from style import *
g=1.0; x0=20.0
t=np.linspace(0,3,400)
fig,ax=plt.subplots(1,2,figsize=(7,2.8))
ks=[0.25,0.7,1.2]
for i,k in enumerate(ks):
    s=np.arctanh(np.sin(k))
    xp=x0*np.cosh(g*t+s)/np.cosh(s); xm=x0*np.cosh(g*t-s)/np.cosh(s)
    ax[0].plot(xp,t,color=COL[i],ls='-',label=fr'$k={k}$')
    ax[0].plot(xm,t,color=COL[i],ls='--')
ax[0].axvspan(10,40,color='#cfcfca',alpha=0.35,lw=0); ax[0].text(25,2.85,r'$A$',ha='center')
ax[0].plot(x0*np.exp(g*t),t,color='#55554f',lw=0.9,ls=':',label=r'null ray $x_0e^{gt}$')
ax[0].set_xscale('log'); ax[0].set_xlim(3,2000); ax[0].set_xlabel(r'$\xi$ (distance from horizon)'); ax[0].set_ylabel(r'$gt$')
ax[0].legend(loc='lower right',frameon=False); ax[0].set_title(r'(a) pair trajectories, $\xi_0=20$',fontsize=9)
# W(k,t) closed form
from qp import W_rind
K=np.linspace(0,np.pi,800)
for i,T in enumerate([0,0.5,1.0,2.0]):
    ax[1].plot(K,W_rind(K,T,16,64)/48,color=COL[i],marker=MK[i],markevery=80,ms=4,label=fr'$gt={T}$')
ax[1].set_yscale('log'); ax[1].set_ylim(1e-3,1.5)
ax[1].set_xlabel(r'$k$'); ax[1].set_ylabel(r'$W_A(k,t)/\ell$'); ax[1].set_title(r'(b) pair weight, $A=[16,64]$, $\rho=0.6$',fontsize=9)
ax[1].set_xticks([0,np.pi/2,np.pi]); ax[1].set_xticklabels(['0',r'$\pi/2$',r'$\pi$'])
ax[1].legend(frameon=False,loc='lower center',ncol=2)
plt.savefig('figs/fig1.pdf'); plt.savefig('figs/fig1.png',dpi=150)
