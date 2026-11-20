import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
def F(R,f): return quad(lambda y:(1-y/R)*f(y),0,R,limit=200)[0]
fs=lambda y: y**2/(1+y**2)**2                       # sin^2Delta/4 in scaling form
fl=lambda y: -2*np.log(max(y**2,1.0)/(1+y**2))      # a(y)
for name,f in [('sigma',fs),('lambda',fl)]:
    r=minimize_scalar(lambda R:-F(R,f)/R,bounds=(0.1,50),method='bounded',options={'xatol':1e-10})
    Rs=r.x; print(name,' R* =',Rs,'  theta*/sqrt(rho) =',np.sqrt(2/Rs), ' (in units of pi:',np.sqrt(2/Rs)/np.pi,')')
# closed form check of F for sigma
R=2.7
I2=0.5*(np.arctan(R)-R/(1+R**2)); I3=0.5*(np.log(1+R**2)+1/(1+R**2)-1)
print('F closed',I2-I3/R,' numeric',F(R,fs))
from phase import sig
for rho in [0.01,0.05,0.1]:
    ths=np.linspace(0.005,0.3,600)*np.pi; S=[sig(t,rho) for t in ths]
    print('rho',rho,' numeric theta*/sqrt(rho)=',ths[np.argmax(S)]/np.sqrt(rho))
