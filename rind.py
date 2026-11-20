import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.signal import lfilter
from gauss import dS2

class Rindler:
    """H = -1/2 sum_{j=1}^{N-1} J_j (c_j^dag c_{j+1} + h.c.), J_j = g*j (or user J)"""
    def __init__(self, N, g=1.0, J=None):
        if J is None: J = g*np.arange(1, N)
        self.N = N
        self.e, self.V = eigh_tridiagonal(np.zeros(N), -0.5*np.asarray(J, float))
    def RA(self, sites, t):
        VA = self.V[list(sites)]
        UA = (VA*np.exp(-1j*self.e*t))@self.V.T        # rows of U for sites in A
        l = len(sites); N = self.N
        R = np.zeros((2*l, 2*N))
        R[0::2, 0::2] = UA.real; R[0::2, 1::2] = UA.imag
        R[1::2, 0::2] = -UA.imag; R[1::2, 1::2] = UA.real
        return R

def apply_gamma_tilted(R, th):
    """returns R @ Gamma0 for the analytic tilted covariance, O(rows*N)"""
    c, s = np.cos(th), np.sin(th)
    Ro, Re = R[:, 0::2], R[:, 1::2]          # columns: odd Majorana (w_{2j-1}) and even (w_{2j})
    out = np.zeros_like(R)
    # on-site: Gamma_{o_j,e_j}=c, Gamma_{e_j,o_j}=-c
    out[:, 1::2] += c*Ro
    out[:, 0::2] += -c*Re
    # (R G)[:, o_j] += sum_{i<j} R[:, e_i] s^2 c^{j-i-1}
    # y_j = sum_{i<j} x_i c^{j-1-i}  -> y = lfilter on shifted x
    x = np.concatenate([np.zeros((R.shape[0], 1)), Re[:, :-1]], axis=1)
    out[:, 0::2] += s**2*lfilter([1.0], [1.0, -c], x, axis=1)
    # (R G)[:, e_i] += -sum_{j>i} R[:, o_j] s^2 c^{j-i-1}  (reverse direction)
    xr = np.concatenate([Ro[:, 1:], np.zeros((R.shape[0], 1))], axis=1)[:, ::-1]
    out[:, 1::2] += -s**2*lfilter([1.0], [1.0, -c], xr, axis=1)[:, ::-1]
    return out

def GA_t(model, sites, t, th):
    R = model.RA(sites, t)
    return apply_gamma_tilted(R, th)@R.T
