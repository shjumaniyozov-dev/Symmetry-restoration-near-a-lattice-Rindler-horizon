import sympy as sp, numpy as np
from scipy.integrate import quad
u,a=sp.symbols('u a',positive=True)
A1,A2,B1,C1,B2,C2=sp.symbols('A1 A2 B1 C1 B2 C2')
ans=A1/(u+1)+A2/(u+1)**2+(B1*u+C1)/(u**2+a**2)+(B2*u+C2)/(u**2+a**2)**2
target=u**2/((u+1)**2*(u**2+a**2)**2)
eqs=sp.Poly(sp.together(ans-target).as_numer_denom()[0],u).coeffs()
sol=sp.solve(eqs,[A1,A2,B1,C1,B2,C2],dict=True)[0]
print({k:sp.factor(v) for k,v in sol.items()}); print('A1+B1=',sp.simplify(sol[A1]+sol[B1]))
I=sol[A1]*sp.log(a)+sol[A2]+sol[C1]*sp.pi/(2*a)+sol[B2]/(2*a**2)+sol[C2]*sp.pi/(4*a**3)
S=sp.simplify(8*a**2/sp.pi*I); print('sigma_hat(theta,1) =',sp.factor(S))
for th in [0.1,0.25,0.4,0.5]:
    av=np.tan(th*np.pi/2)**2; c=np.cos(th*np.pi)
    num=quad(lambda k: 2/(1+np.sin(k))*np.sin(th*np.pi)**4*np.sin(k)**2/(1+c**2-2*c*np.cos(k))**2,0,np.pi)[0]/(2*np.pi)
    print(th, float(S.subs(a,av)) if th!=0.5 else float(sp.limit(S,a,1)), num)
dS=sp.simplify(sp.diff(S,a)); print('dsigma/da =',sp.factor(dS))
fa=sp.lambdify(a,dS,'numpy'); aa=np.linspace(1e-4,1-1e-4,200001)
print('min dsigma/da on (0,1):',np.min(fa(aa)))
