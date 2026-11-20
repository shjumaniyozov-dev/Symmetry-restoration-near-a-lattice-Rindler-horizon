import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
def sin2D(k,th):
    c=np.cos(th); return np.sin(th)**4*np.sin(k)**2/(1+c**2-2*c*np.cos(k))**2
def a_k(k,th):
    c=np.cos(th); cd=np.abs((2*c-(1+c**2)*np.cos(k))/(1+c**2-2*c*np.cos(k))); return -2*np.log((1+cd)/2)
def wlate(k,rho):   # 2[1-sin k/rho]_+/cos^2 k   (finite at k=pi/2 when rho=1)
    s=np.sin(k)
    if rho>=1: return 2*(1-s/rho)/(1-s**2) if s<1 else 1.0
    return 2*max(1-s/rho,0)/(1-s**2)
def coef(th,rho,f):
    kmax=np.pi/2 if rho>=1 else np.arcsin(rho)
    I=quad(lambda k: wlate(k,rho)*f(k,th),0,kmax,limit=300,points=None)[0]
    return 2*I/(2*np.pi)          # k in (0,kmax) and (pi-kmax,pi): f symmetric? no -> do both sides
def coef_full(th,rho,f):
    kmax=np.pi/2 if rho>=1 else np.arcsin(rho)
    I1=quad(lambda k: wlate(k,rho)*f(k,th),0,kmax,limit=400)[0]
    I2=quad(lambda k: wlate(k,rho)*f(k,th),np.pi-kmax,np.pi,limit=400)[0]
    return (I1+I2)/(2*np.pi)
sig=lambda th,rho: coef_full(th,rho,sin2D)
lam=lambda th,rho: coef_full(th,rho,a_k)
# checks: rho->0 asymptotics
for th in [0.2*np.pi,0.4*np.pi]:
    c=np.cos(th); AB=2*(1+6*c**2+c**4)/(1-c**2)**2
    for rho in [0.05,0.01]:
        print(f"th={th/np.pi:.1f}pi rho={rho}: sigma/rho^3={sig(th,rho)/rho**3:.5f}  (A+B)/(12pi)={AB/(12*np.pi):.5f}")
print('t=0 check: int_0^pi dk/2pi sin2D =',quad(lambda k: sin2D(k,0.3*np.pi),0,np.pi)[0]/(2*np.pi),' sin^2/4=',np.sin(0.3*np.pi)**2/4)
for rho in [1.0,0.9,0.8,0.7,0.6,0.4,0.2]:
    print('rho',rho,' sigma(th):',' '.join(f"{sig(th*np.pi,rho):.4f}" for th in [0.1,0.2,0.3,0.4,0.5]),
          ' | lambda(th):',' '.join(f"{lam(th*np.pi,rho):.4f}" for th in [0.1,0.2,0.3,0.4,0.5]))
