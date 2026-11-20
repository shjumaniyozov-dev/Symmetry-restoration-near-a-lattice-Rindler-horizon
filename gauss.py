"""Fermionic Gaussian-state tools (Majorana covariance), conventions:
 w_{2j-1}=c_j+c_j^dag, w_{2j}=i(c_j-c_j^dag)  (0-based: w[2j], w[2j+1])
 <w_a w_b> = delta_ab + i Gamma_ab ;  Gamma_{2j-1,2j} = <sigma^z_j> = 2n_j-1
"""
import numpy as np

def gamma_tilted(N, th):
    """Majorana covariance of the fermionic (cat) tilted ferromagnet,
    exact product-state two-point functions (cross terms O(cos^N th) dropped)."""
    c, s = np.cos(th), np.sin(th)
    G = np.zeros((2*N, 2*N))
    for j in range(N):
        G[2*j, 2*j+1] = c
    for i in range(N):
        for j in range(i+1, N):
            G[2*i+1, 2*j] = s**2 * c**(j-i-1)
    return G - G.T

def hop_matrix(J):
    """single particle h for H=-1/2 sum_j J_j (c_j^dag c_{j+1}+h.c.)"""
    N = len(J)+1
    h = np.zeros((N, N))
    for j, Jj in enumerate(J):
        h[j, j+1] = h[j+1, j] = -0.5*Jj
    return h

class Evolver:
    def __init__(self, h):
        self.e, self.V = np.linalg.eigh(h)
    def R(self, t):
        U = (self.V*np.exp(-1j*self.e*t))@self.V.T   # c(t)=U c
        Ur, Ui = U.real, U.imag
        N = U.shape[0]
        R = np.zeros((2*N, 2*N))
        R[0::2, 0::2] = Ur; R[0::2, 1::2] = Ui
        R[1::2, 0::2] = -Ui; R[1::2, 1::2] = Ur
        return R
    def evolve(self, G, t):
        R = self.R(t); return R@G@R.T

def restrict(G, sites):
    idx = np.ravel([[2*j, 2*j+1] for j in sites])
    return G[np.ix_(idx, idx)]

def rot(GA, a):
    """Gamma of e^{i a Q} rho e^{-i a Q}: c -> e^{i a} c in expectation values"""
    n = GA.shape[0]//2
    R = np.zeros_like(GA)
    ca, sa = np.cos(a), np.sin(a)
    for j in range(n):
        R[2*j:2*j+2, 2*j:2*j+2] = [[ca, sa], [-sa, ca]]
    return R@GA@R.T

def trprod(G1, G2):
    """Tr(rho1 rho2) = sqrt(det((1-G1 G2)/2)) (positive branch)"""
    M = (np.eye(G1.shape[0]) - G1@G2)/2
    sgn, ld = np.linalg.slogdet(M)
    return np.exp(0.5*ld)

def dS2(GA, M=None):
    """Renyi-2 entanglement asymmetry, exact alpha quadrature (trig polynomial)"""
    l = GA.shape[0]//2
    if M is None: M = 2*l+4
    al = 2*np.pi*np.arange(M)/M
    den = trprod(GA, GA)
    num = np.mean([trprod(GA, rot(GA, a)) for a in al])
    return -np.log(num/den)

def xy_majorana_M(N, gam, h, hedge=None):
    """Real antisymmetric M with H = (i/4) sum_ab M_ab w_a w_b for the fermionic
    open XY chain  H = sum_j [ -(1+gam)/2 X_j X_{j+1} - (1-gam)/2 Y_j Y_{j+1} ] - sum_j h_j Z_j,
    written via JW:  Z_j = -i w_{2j-1} w_{2j},  X_j X_{j+1} = -i w_{2j} w_{2j+1},
    Y_j Y_{j+1} = +i w_{2j-1} w_{2j+2}."""
    if hedge is None: hedge = h/2
    hs = np.full(N, h); hs[0] = hs[-1] = hedge
    M = np.zeros((2*N, 2*N))
    # H = (i/4) sum M_ab w_a w_b = (i/2) sum_{a<b} M_ab w_a w_b
    def add(a, b, coef):   # adds coef * (i w_a w_b) with a<b  -> M_ab += 2*coef
        M[a, b] += 2*coef; M[b, a] -= 2*coef
    for j in range(N):
        add(2*j, 2*j+1, +hs[j])            # -h Z = -h(-i w w) = +h (i w w)
    for j in range(N-1):
        add(2*j+1, 2*j+2, +(1+gam)/2)      # -(1+g)/2 XX = -(1+g)/2(-i w w)
        add(2*j, 2*j+3, -(1-gam)/2)        # -(1-g)/2 YY = -(1-g)/2(+i w w)
    return M

def ground_gamma(M, ref=None, tol=1e-9):
    """Gamma_GS = -i sgn(iM); exact zero modes resolved as real pair (a,b),
    sign chosen to be closest to reference covariance `ref`."""
    lam, V = np.linalg.eigh(1j*M)
    nz = np.abs(lam) > tol
    G = (-1j*(V[:, nz]*np.sign(lam[nz]))@V[:, nz].conj().T).real
    if (~nz).sum() == 2:
        K = V[:, ~nz]
        # real basis of kernel
        Kr = np.linalg.qr(np.hstack([K.real, K.imag]))[0]
        u, s, vt = np.linalg.svd(np.hstack([K.real, K.imag]), full_matrices=False)
        a, b = u[:, 0], u[:, 1]
        Z = np.outer(a, b) - np.outer(b, a)
        cands = [G+Z, G-Z]
        if ref is not None:
            G = min(cands, key=lambda X: np.abs(X-ref).sum())
        else:
            G = cands[0]
    return G
