import numpy as np
kg, kw = np.polynomial.legendre.leggauss(4000)
K = (kg+1)*np.pi/2; KW = kw*np.pi/2/(2*np.pi)        # \int_0^pi dk/2pi
bg, bw = np.polynomial.legendre.leggauss(800)
B = bg*np.pi; BW = bw*np.pi/(2*np.pi)                # \int_-pi^pi dbeta/2pi
def sin2D(k, th):
    c=np.cos(th); return np.sin(th)**4*np.sin(k)**2/(1+c**2-2*c*np.cos(k))**2
def absCosD(k, th):
    c=np.cos(th); return np.abs((2*c-(1+c**2)*np.cos(k))/(1+c**2-2*c*np.cos(k)))
def W_hom(k, t, l, J=1.0):
    return np.maximum(l-2*J*np.abs(np.sin(k))*t, 0)
def W_rind(k, t, xa, xb, g=1.0):
    l=xb-xa; xbar=(xa+xb)/2
    ch=np.cosh(g*t)
    return np.maximum(l-2*xbar*np.tanh(g*t)*np.abs(np.sin(k)),0)*ch/(ch**2*np.cos(k)**2+np.sin(k)**2)
_LOG = None
def dS2_qp(Wk, th):
    s2=sin2D(K,th)
    L=np.log(np.clip(1-np.outer(np.sin(B/2)**2,s2),1e-300,None))   # (beta,k)
    expo=L@(KW*Wk)
    return -np.log(np.sum(BW*np.exp(expo)))
def Sigma(Wk, th): return np.sum(KW*Wk*sin2D(K,th))
def lin(Wk, th): return np.sum(KW*Wk*(-2*np.log((1+absCosD(K,th))/2)))
def W_rind_cut(k, t, xa, xb, xc, g=1.0):
    s=np.arctanh(np.clip(np.abs(np.sin(k)),0,1-1e-15))
    lo=np.maximum(xa*np.cosh(s)/np.cosh(g*t-s), xc)
    hi=xb*np.cosh(s)/np.cosh(g*t+s)
    return np.maximum(hi-lo,0)
