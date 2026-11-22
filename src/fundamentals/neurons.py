import numpy as np
def mcp_neuron(inputs,weights,threshold): return int(np.dot(np.asarray(inputs),np.asarray(weights))>=threshold)
def sigmoid(x): return 1/(1+np.exp(-np.asarray(x,dtype=float)))
class TinyMLP:
    """Single-hidden-layer sigmoid MLP with explicit forward/backprop."""
    def __init__(self,n_in=2,n_hidden=4,n_out=1,seed=7):
        r=np.random.default_rng(seed); self.W1=r.normal(0,.5,(n_in,n_hidden)); self.b1=np.zeros(n_hidden)
        self.W2=r.normal(0,.5,(n_hidden,n_out)); self.b2=np.zeros(n_out)
    def forward(self,x):
        self.x=np.asarray(x,float); self.h=sigmoid(self.x@self.W1+self.b1); self.y=sigmoid(self.h@self.W2+self.b2); return self.y
    def backward(self,target,lr=.1):
        t=np.asarray(target,float).reshape(self.y.shape); d2=(self.y-t)*self.y*(1-self.y)
        gW2=self.h.T@d2; gb2=d2.sum(0); d1=(d2@self.W2.T)*self.h*(1-self.h)
        self.W2-=lr*gW2; self.b2-=lr*gb2; self.W1-=lr*(self.x.T@d1); self.b1-=lr*d1.sum(0)
        return float(np.mean((self.y-t)**2))
