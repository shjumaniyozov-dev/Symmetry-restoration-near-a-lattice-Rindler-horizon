import numpy as np
from scipy.optimize import minimize_scalar
from style import *
import phase
from phase import sig, lam
def thstar(f,rho):
    r=minimize_scalar(lambda th:-f(th,rho),bounds=(1e-3,np.pi/2),method='bounded',options={'xatol':1e-7})
    return r.x
rhos=np.concatenate([np.linspace(0.005,0.9,40),1-np.logspace(-1,-3.3,18)])
TS=np.array([thstar(sig,r) for r in rhos]); TL=np.array([thstar(lam,r) for r in rhos])
np.savez('phase_curve.npz',rhos=rhos,TS=TS,TL=TL)
fig,ax=plt.subplots(1,2,figsize=(7,2.9))
ax[0].fill_between(rhos,TS/np.pi,0.5,color=COL[0],alpha=0.12,lw=0)
ax[0].plot(rhos,TS/np.pi,color=COL[0],label=r'$\partial_\theta\hat\sigma=0$ (Gaussian window)')
ax[0].plot(rhos,TL/np.pi,color=COL[1],ls='--',label=r'$\partial_\theta\hat\lambda=0$ (linear regime)')
rr=np.linspace(0,0.3,50); ax[0].plot(rr,0.83141*np.sqrt(rr)/np.pi,color='#55554f',ls=':',lw=1,label=r'$0.8314\sqrt{\rho}$')
ax[0].text(0.72,0.36,'Mpemba\n(crossing)',ha='center',fontsize=8); ax[0].text(0.75,0.08,'no crossing',ha='center',fontsize=8)
ax[0].set_xlabel(r'$\rho=\tanh(g\,\ell_{\rm opt}/2)$'); ax[0].set_ylabel(r'$\theta^*/\pi$'); ax[0].set_xlim(0,1); ax[0].set_ylim(0,0.5)
ax[0].legend(frameon=False,loc='upper left',fontsize=7); ax[0].set_title('(a) boundary of the Mpemba region',fontsize=9)
th=np.linspace(0.005,0.5,200)*np.pi
c=np.cos(th); sn=np.sin(th)
closed=sn**2*(4-np.pi+6*np.pi*c**2-(4+np.pi)*c**4-16*c*sn**2*np.arctanh(np.minimum(c,1-1e-15)))/(np.pi*(1+c**2)**3)
for i,rho in enumerate([1.0,0.95,0.8,0.5]):
    v=np.array([sig(t,rho) for t in th[::8]])
    ax[1].plot(th[::8]/np.pi,v/np.max(v),color=COL[i],marker=MK[i],ms=3.5,label=fr'$\rho={rho}$')
ax[1].plot(th/np.pi,closed/closed.max(),color='k',lw=0.8,label=r'closed form, $\rho=1$')
ax[1].set_xlabel(r'$\theta/\pi$'); ax[1].set_ylabel(r'$\hat\sigma(\theta,\rho)/\max_\theta\hat\sigma$')
ax[1].legend(frameon=False,fontsize=7,loc='lower right'); ax[1].set_title(r'(b) late-time amplitude',fontsize=9)
plt.savefig('figs/fig4.pdf'); plt.savefig('figs/fig4.png',dpi=150)
print('closed vs numeric at rho=1 max rel err:',np.max(np.abs(np.array([sig(t,1.0) for t in th[::10]])-closed[::10])/closed[::10]))
for r,a,b in zip(rhos[::6],TS[::6],TL[::6]): print(f'{r:.4f} {a/np.pi:.4f} {b/np.pi:.4f}')
