"""Independent re-verification of every formula/number quoted in the manuscript."""
import numpy as np, mpmath as mp
from scipy.integrate import quad
from scipy.optimize import brentq
from qp import *
from phase import sig, lam, sin2D, a_k
ok=lambda b: 'OK' if b else 'FAIL'
# Lemma 1
for th in [0.15*np.pi,0.33*np.pi,0.5*np.pi]:
  for al in [0.3,1.2,2.0,2.9]:
    lhs=quad(lambda k: np.log(1-np.sin(al)**2*sin2D(k,th)),0,np.pi,limit=400)[0]/(2*np.pi)
    rhs=np.log((1+np.cos(th)**2+np.sin(th)**2*abs(np.cos(al)))/2)
    assert abs(lhs-rhs)<1e-9,(th,al,lhs,rhs)
print('Lemma 1 (Jensen identity), 12 checks:',ok(True))
# sin^2 Delta closed form vs Ares cosDelta
for th in [0.2,0.4]:
  for k in [0.3,1.1,2.5]:
    c=np.cos(th*np.pi); cd=(2*c-(1+c**2)*np.cos(k))/(1+c**2-2*c*np.cos(k))
    assert abs(1-cd**2-sin2D(k,th*np.pi))<1e-14
print('sin^2 Delta_k closed form:',ok(True))
# t=0 asymptotics
for th in [0.2*np.pi,0.5*np.pi]:
  for l in [48,192,2000]:
    ex=-np.log(quad(lambda a:(1-np.sin(th)**2*np.sin(a/2)**2)**l,-np.pi/2,np.pi/2,limit=400,points=[0])[0]/np.pi)
    print(f'  Delta S(0) th={th/np.pi:.1f}pi l={l}: exact {ex:.5f}  1/2 ln(pi l sin^2/4) {0.5*np.log(np.pi*l*np.sin(th)**2/4):.5f}')
# W properties
for (xa,xb) in [(0,48),(16,64)]:
  for k in [0.4,1.3]:
    l=xb-xa; xbar=(xa+xb)/2
    assert abs(W_rind(np.array([k]),0,xa,xb)[0]-l)<1e-12
    h=1e-6; dW=(W_rind(np.array([k]),h,xa,xb)[0]-W_rind(np.array([k]),0,xa,xb)[0])/h
    assert abs(dW+2*xbar*np.sin(k))<1e-4
    T=12.0; late=2*l*np.exp(-T)*max(1-np.sin(k)/(l/(2*xbar)),0)/np.cos(k)**2
    Wt=W_rind(np.array([k]),T,xa,xb)[0]; assert (late==0 and Wt==0) or abs(Wt/late-1)<1e-6
# homogeneous limit g->0
g=1e-5; J=3.0; L=40; xbar=J/g
Wg=W_rind(np.array([0.7]),5.0,xbar-L/2,xbar+L/2,g=g)[0]; print('  g->0 limit W:',Wg,' homogeneous:',max(L-2*J*np.sin(0.7)*5,0))
print('W(k,t) properties (t=0, slope, late-time, g->0):',ok(abs(Wg-max(L-2*J*np.sin(0.7)*5,0))<1e-3))
# closed form sigma(theta,1)
def sclosed(th):
    c=np.cos(th); s=np.sin(th)
    return s**2*(4-np.pi+6*np.pi*c**2-(4+np.pi)*c**4-16*c*s**2*np.arctanh(c))/(np.pi*(1+c**2)**3)
err=max(abs(sclosed(t)/sig(t,1.0)-1) for t in np.linspace(0.02,0.5,25)*np.pi)
print('closed form sigma(theta,1) max rel err:',err, ok(err<1e-7), '  sigma(pi/2,1)=',sig(np.pi/2,1.0),'(4-pi)/pi=',(4-np.pi)/np.pi)
print('  small theta: sigma/(theta^2/2) at theta=0.01:',sclosed(0.01)/(0.01**2/2))
# monotonicity of lambda(theta,1)
ths=np.linspace(0.005,0.5,300)*np.pi; L1=np.array([lam(t,1.0) for t in ths]); S1=np.array([sig(t,1.0) for t in ths])
print('lambda(theta,1) monotone increasing on grid:',ok(np.all(np.diff(L1)>0)),' sigma(theta,1):',ok(np.all(np.diff(S1)>0)))
print('  lambda(pi/2,1)=',lam(np.pi/2,1.0))
# small-rho
f=lambda R: R*(np.arctan(R)-R/(1+R**2))-2*(np.log(1+R**2)+1/(1+R**2)-1)
Rs=brentq(f,1,5); print('R* =',Rs,' sqrt(2/R*)=',np.sqrt(2/Rs))
for th in [0.25*np.pi,0.45*np.pi]:
    c=np.cos(th); pred=(1+6*c**2+c**4)/(6*np.pi*np.sin(th)**4)
    print(f'  small rho th={th/np.pi}pi: sigma/rho^3 (rho=0.005) {sig(th,0.005)/0.005**3:.6f}  predicted {pred:.6f}')
# Gaussian-state t=0 vs formula, and Gaussian charged moment formula vs QP at t=0
from gauss import gamma_tilted, restrict, dS2
for th in [0.2*np.pi,0.4*np.pi]:
    l=48; G=restrict(gamma_tilted(300,th),range(100,100+l))
    ex=-np.log(quad(lambda a:(1-np.sin(th)**2*np.sin(a/2)**2)**l,-np.pi,np.pi,limit=400,points=[0])[0]/(2*np.pi)/(0.5*(1+np.cos(th)**(2*l))))
    print(f'  t=0 Gaussian dS2 {dS2(G):.10f}  exact-mixture formula {ex:.10f}  QP {dS2_qp(np.full_like(K,l),th):.10f}')
