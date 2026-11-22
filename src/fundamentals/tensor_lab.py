import numpy as np
def tensor_examples():
    a=np.arange(12).reshape(3,4); b=np.array([1,2,3,4])
    return {"matrix":a,"broadcast_sum":a+b,"inner_product":float(b@b),"reshaped":a.reshape(2,6),"svd":np.linalg.svd(a,full_matrices=False)}
