import numpy as np
def optimize(method="sgd",steps=80,lr=.08,seed=0):
    x=np.array([4.,-3.]); v=np.zeros(2); hist=[]
    for i in range(steps):
        grad=np.array([2*x[0],10*x[1]])
        if method=="batch_gd": update=grad
        elif method=="sgd": update=grad+np.random.default_rng(seed+i).normal(0,.25,2)
        elif method=="momentum": v=.9*v+grad; update=v
        elif method=="nesterov":
            look=x-.9*v; v=.9*v+np.array([2*look[0],10*look[1]]); update=v
        else: raise ValueError(method)
        x-=lr*update; hist.append(float(x[0]**2+5*x[1]**2))
    return np.asarray(hist)
def numerical_gradient(fn,x,eps=1e-5):
    x=np.asarray(x,float); g=np.zeros_like(x)
    for i in range(len(x)):
        a=x.copy(); b=x.copy(); a[i]+=eps; b[i]-=eps; g[i]=(fn(a)-fn(b))/(2*eps)
    return g
