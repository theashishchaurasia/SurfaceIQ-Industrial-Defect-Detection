import numpy as np
def hebbian_update(w,x,y,lr=.1): return w+lr*y*np.asarray(x)
def modified_hebbian_update(w,x,y,lr=.1,decay=.01): return w+lr*y*np.asarray(x)-decay*w
def competitive_learning(weights,x,lr=.1):
    w=np.asarray(weights,float); winner=np.argmin(np.linalg.norm(w-x,axis=1)); w[winner]+=lr*(np.asarray(x)-w[winner]); return w,winner
def error_correction(w,x,target,lr=.1):
    pred=float(np.dot(w,x)); return w+lr*(target-pred)*np.asarray(x),pred
def energy_binary(state,weights,bias=None):
    s=np.asarray(state,float); b=np.zeros_like(s) if bias is None else np.asarray(bias)
    return float(-.5*s@weights@s-b@s)
