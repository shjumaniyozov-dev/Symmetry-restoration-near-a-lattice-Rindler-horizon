import sympy as sp, numpy as np
a=sp.symbols('a',positive=True)
S=2*a*(sp.pi*a**4-8*a**3*sp.log(a)+8*a**3-6*sp.pi*a**2+8*a*sp.log(a)+8*a+sp.pi)/(sp.pi*(a**2+1)**3)
# symmetry a->1/a (theta -> pi - theta)
print('S(1/a)-S(a)=',sp.simplify(S.subs(a,1/a)-S))
P=-(sp.pi*a**6-16*a**5*sp.log(a)+24*a**5-23*sp.pi*a**4+64*a**3*sp.log(a)+23*sp.pi*a**2-16*a*sp.log(a)-24*a-sp.pi)
print('series of P at a=1:',sp.series(P,a,1,5))
# in theta variable: a=tan^2(theta/2)
th=sp.symbols('theta',positive=True)
St=sp.simplify(S.subs(a,sp.tan(th/2)**2))
print('S(theta) =',St)
f=sp.lambdify(th,sp.diff(St,th),'mpmath')
import mpmath as mp
mp.mp.dps=30
vals=[f(mp.mpf(x)) for x in mp.linspace(mp.mpf('1e-6'),mp.pi/2-mp.mpf('1e-6'),20001)]
print('min dS/dtheta on (0,pi/2) [30-digit, 20001 pts]:',min(vals))
