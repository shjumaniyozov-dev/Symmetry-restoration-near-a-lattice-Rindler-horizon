import numpy as np
from style import *
from qp import *
from phase import sig
d48=np.load('rind_N6000.npz'); d96=np.load('rind_L96.npz'); d192=np.load('rind_L192.npz')
ths=d192['ths']; tf=np.linspace(0,4,161)
def qpcurve(xa,xb,th,ts): return np.array([dS2_qp(W_rind(K,t,xa,xb),th) for t in ts])
# ---------- Fig 2: horizon-adjacent
fig,ax=plt.subplots(1,2,figsize=(7.2,2.9)); plt.subplots_adjust(wspace=0.38)
ts=d192['ts']; m=ts<=3.6+1e-9
for i,th in enumerate(ths):
    ax[0].plot(ts[m],d192['res'][0,i,m],ls='none',marker=MK[i],ms=4,color=COL[i],label=fr'$\theta={th/np.pi:.1f}\pi$')
    ax[0].plot(tf[tf<=3.6],qpcurve(0,192,th,tf[tf<=3.6]),color=COL[i],lw=1.2)
tt=np.linspace(1,3.6,10); s=sig(np.pi/2,1.0)
ax[0].plot(tt,0.5*np.log(np.pi*192*s)-tt/2+0.25,color='#55554f',ls=':',lw=1)
ax[0].text(2.55,1.75,r'slope $-g/2$',fontsize=8,color='#55554f')
ax[0].set_xlabel(r'$gt$'); ax[0].set_ylabel(r'$\Delta S_A^{(2)}$'); ax[0].legend(frameon=False,loc='lower left')
ax[0].set_title(r'(a) $A=[0,192]$ ($\rho=1$): no crossing',fontsize=9)
th=np.pi/2
for j,(dd,l,tmax) in enumerate([(d48,48,4.0),(d96,96,3.4),(d192,192,3.6)]):
    ts=dd['ts']; m=ts<=tmax+1e-9
    ax[1].plot(ts[m],dd['res'][0,3,m]-0.5*np.log(l),ls='none',marker=MK[j],ms=4,color=COL[j],label=fr'lattice, $\ell={l}$')
ax[1].plot(tf,qpcurve(0,192,th,tf)-0.5*np.log(192),color=COL[2],lw=1.2,label=r'QP, $\ell=192$')
ax[1].plot(tf,qpcurve(0,48,th,tf)-0.5*np.log(48),color=COL[0],lw=1.2,ls='--',label=r'QP, $\ell=48$')
ax[1].plot(tf,0.5*np.log(np.pi*s)-tf/2,color='#55554f',ls=':',lw=1,label=r'$\frac{1}{2}\ln(\pi\hat\sigma)-\frac{gt}{2}$')
ax[1].set_xlabel(r'$gt$'); ax[1].set_ylabel(r'$\Delta S_A^{(2)}-\frac{1}{2}\ln\ell$'); ax[1].set_ylim(-2.6,0.8)
ax[1].legend(frameon=False,fontsize=7,loc='lower left'); ax[1].set_title(r'(b) $\theta=\pi/2$, scaling with $\ell$',fontsize=9)
plt.savefig('figs/fig2.pdf'); plt.savefig('figs/fig2.png',dpi=150)
# ---------- Fig 3: away from the horizon
fig,ax=plt.subplots(1,2,figsize=(7.2,2.9)); plt.subplots_adjust(wspace=0.38)
for p,(dd,gi,xa,xb,tmax,title) in enumerate([(d192,1,64,256,3.6,r'(a) $A=[64,256]$ ($\rho=0.6$)'),(d96,2,96,192,3.2,r'(b) $A=[96,192]$ ($\rho=1/3$)')]):
    ts=dd['ts']; m=ts<=tmax+1e-9; tq=tf[tf<=tmax]
    for i,th in enumerate(ths):
        ax[p].plot(ts[m],dd['res'][gi,i,m],ls='none',marker=MK[i],ms=4,color=COL[i],label=fr'$\theta={th/np.pi:.1f}\pi$')
        ax[p].plot(tq,qpcurve(xa,xb,th,tq),color=COL[i],lw=1.2)
    ax[p].set_yscale('log'); ax[p].set_ylim(1e-2,3); ax[p].set_xlabel(r'$gt$'); ax[p].set_ylabel(r'$\Delta S_A^{(2)}$'); ax[p].set_title(title,fontsize=9)
ax[0].legend(frameon=False,loc='lower left')
plt.savefig('figs/fig3.pdf'); plt.savefig('figs/fig3.png',dpi=150)
