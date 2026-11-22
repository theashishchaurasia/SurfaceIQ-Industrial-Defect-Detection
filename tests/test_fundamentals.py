import numpy as np
from src.fundamentals.neurons import mcp_neuron,TinyMLP
from src.fundamentals.optimizers import numerical_gradient,optimize
from src.fundamentals.learning_paradigms import energy_binary
def test_neurons():
    assert mcp_neuron([1,1],[1,1],2)==1
    m=TinyMLP(); assert m.forward([[0,1],[1,0]]).shape==(2,1)
    assert m.backward([[1],[1]])>=0
def test_gradient():
    np.testing.assert_allclose(numerical_gradient(lambda x:float((x*x).sum()),[2.,-3.]),[4.,-6.],atol=1e-4)
def test_optimizers():
    for n in ["batch_gd","sgd","momentum","nesterov"]: assert np.isfinite(optimize(n,10)).all()
def test_energy(): assert np.isfinite(energy_binary([1,-1],np.eye(2)))
