# not realy tested
import biord # en eigen
from scipy import integrate
import numpy as np

m = biorbd.Model("SliderContact.bioMod")
tau = np.zeros((m.nbQ()))
tau[1]=1

def fd(t, x):
    q = x[m.nbQ(), :]
    dq = x[m.nbQ(), :]
    ddq = m.ForwardDyanmics(q, dq, tau).to_array()
    return np.concatenate((dq, ddq))

scipy.integrate.solve_ivp(fd,[0,5],np.zeros((m.nbQ*2,)), t_eval= np.linspace(0,5,100))