import numpy as np
def activation_curves(x):
    x=np.asarray(x,float); return {"sigmoid":1/(1+np.exp(-x)),"tanh":np.tanh(x),"relu":np.maximum(0,x)}
