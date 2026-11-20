import sympy as sp, numpy as np
t,g,s,xa,xb=sp.symbols('t g s xi_a xi_b',positive=True)
W1=sp.cosh(s)*(xb/sp.cosh(g*t+s)-xa/sp.cosh(g*t-s))
l=xb-xa; xbar=(xa+xb)/2
W2=sp.cosh(g*t)*sp.cosh(s)**2*(l-2*xbar*sp.tanh(g*t)*sp.tanh(s))/(sp.cosh(g*t)**2+sp.sinh(s)**2)
print('W1-W2 =',sp.simplify((W1-W2).rewrite(sp.exp)))
# in terms of k: tanh s = sin k, cosh^2 s = 1/cos^2 k, sinh^2 s = tan^2 k
k=sp.symbols('k',positive=True)
W3=sp.cosh(g*t)*(l-2*xbar*sp.tanh(g*t)*sp.sin(k))/(sp.cosh(g*t)**2*sp.cos(k)**2+sp.sin(k)**2)
print('W2-W3 =',sp.simplify((W2.subs(s,sp.atanh(sp.sin(k)))-W3)))
print('dW/dt at 0 =',sp.simplify(sp.diff(W3,t).subs(t,0)))
print('g->0 limit with xbar=J/g:',sp.limit(W3.subs({xa:sp.Symbol('J')/g-l/2}).subs(xb,sp.Symbol('J')/g+sp.Symbol('L')/2),g,0))
# sin^2 Delta closed form
c,kk=sp.symbols('c k',real=True)
cosD=(2*c-(1+c**2)*sp.cos(kk))/(1+c**2-2*c*sp.cos(kk))
print('1-cosD^2 - (1-c^2)^2 sin^2k/(1+c^2-2c cos k)^2 =',sp.simplify(1-cosD**2-(1-c**2)**2*sp.sin(kk)**2/(1+c**2-2*c*sp.cos(kk))**2))
