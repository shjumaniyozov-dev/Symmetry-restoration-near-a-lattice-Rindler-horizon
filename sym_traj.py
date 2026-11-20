import sympy as sp
t,g,x0,s0=sp.symbols('t g x0 s0',real=True)
# claimed solution of Hamilton eqs for eps=-J(x) cos k, J=g x:
#  xdot = J sin k, kdot = J'(x) cos k   (xdot=d eps/dk, kdot=-d eps/dx)
s=s0+g*t
sink=sp.tanh(s); cosk=1/sp.cosh(s)   # branch cos k>0
x=x0*sp.cosh(s)/sp.cosh(s0)
k=sp.asin(sp.tanh(s))
eq1=sp.simplify(sp.diff(x,t)-g*x*sink)
eq2=sp.simplify(sp.diff(k,t)-g*sp.cos(k))
print('xdot residual',eq1,' kdot residual',sp.simplify(eq2.rewrite(sp.exp)))
# energy conservation
E=-g*x*sp.cos(k)
print('dE/dt',sp.simplify(sp.diff(E,t)))
# branch cos k<0: k=pi-asin(tanh s)
k2=sp.pi-sp.asin(sp.tanh(s)); x2=x0*sp.cosh(s)/sp.cosh(s0)
print('branch2 xdot',sp.simplify(sp.diff(x2,t)-g*x2*sp.sin(k2)),' kdot',sp.simplify((sp.diff(k2,t)-g*sp.cos(k2)).rewrite(sp.exp)))
# pair: +s0 and -s0, separation and centre
xp=x0*sp.cosh(g*t+s0)/sp.cosh(s0); xm=x0*sp.cosh(g*t-s0)/sp.cosh(s0)
print('sep',sp.simplify(sp.expand_trig((xp-xm)).rewrite(sp.exp)), ' -> expected 2 x0 sinh(gt) tanh(s0):',sp.simplify((xp-xm-2*x0*sp.sinh(g*t)*sp.tanh(s0)).rewrite(sp.exp)))
print('centre-x0cosh(gt)',sp.simplify(((xp+xm)/2-x0*sp.cosh(g*t)).rewrite(sp.exp)))
