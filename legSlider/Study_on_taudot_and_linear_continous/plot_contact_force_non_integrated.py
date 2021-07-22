import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from bioptim import *
from biorbd import *

cur_path = '/examples_by_me/legSlider/Study_on_taudot_and_linear_continous/'
files = ['tau.bo', "tau_LC.bo", 'dtau.bo', "dtau_LC.bo"]
label = ['Tau driven', 'Tau driven PiecewiseLinear', 'Taudot driven', 'Taudot driven PiecewiseLinear']

sol = []
soli = []
all_ocp = []
for i, v in enumerate(files):
    ocp, solution = OptimalControlProgram.load(cur_path + v)
    sol.append(solution)
    soli.append(solution.integrate(shooting_type=Shooting.MULTIPLE, keepdims=False))
    all_ocp.append(ocp)


def Compute_all_contact_forces(ocp, sol):
    """
    """
    X = sol.states["all"]
    U = sol.controls["all"]
    P = sol.parameters['all']
    forces = np.zeros((2, sol.ns[0] + 1))
    for ii in range(sol.ns[0] + 1):
        forces[[0, 1], ii] = np.squeeze(
            ocp.nlp[0].contact_forces_func(X[:, ii], U[:, ii], P).toarray())
    return forces

forces = np.zeros((2, 21, 4))
for jj in range(len(files)):
    forces[:, :, jj] = Compute_all_contact_forces(all_ocp[jj], sol[jj])

fig, axs = plt.subplots(2, figsize=(19.2, 10.8))
X = np.linspace(sol[i].phase_time[0], sol[i].phase_time[1], sol[i].ns[0] + 1)
for jj in range(len(files)):
    for ii in range(2):
        Y = forces[ii, :, jj]
        axs[ii].plot(X, Y, 'o-', ms=2,  label=label[jj])
        axs[ii].set(xlabel='Time (s)', ylabel='Force (N)')

plt.tight_layout()
plt.legend()
plt.savefig(cur_path + "contact_forces")
