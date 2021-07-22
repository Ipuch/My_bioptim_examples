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

    # compute controls for integrated solution
    steps = soli[0].ocp.nlp[0].ode_solver.steps
    ns = sol[i].ns[0]
    ns_int = ns * steps + 1

    U = dict()
    if soli[i].ocp.nlp[0].control_type.name == "CONSTANT":
        for ii, var in enumerate(sol[i].controls.keys()):
            U[var] = np.zeros((sol[i].controls[var].shape[0],
                               ns_int))
            idx = list(range(ns_int))
            U[var][:, 0:ns_int:5] = sol[i].controls[var]

            for kk in range(1, steps):
                U[var][:, idx[kk:ns_int:steps]] = sol[i].controls[var][:, :-1]
        # soli[i].controls = U  # fill the controls field
        soli[i].icontrols = U

    elif soli[i].ocp.nlp[0].control_type.name == "LINEAR_CONTINUOUS":
        x = np.linspace(sol[i].phase_time[0], sol[i].phase_time[1], ns + 1)
        X = np.linspace(sol[i].phase_time[0], sol[i].phase_time[1], ns_int)
        for ii, var in enumerate(sol[i].controls.keys()):
            f = interp1d(x, sol[i].controls[var])
            U[var] = f(X)
        # soli[i].controls = U  # fill the controls field
        soli[i].icontrols = U
    soli[i].parameters = sol[i].parameters


def Compute_all_contact_forces(ocp, sol):
    """
    """
    X = sol.states["all"]
    # U = sol.controls["all"]
    U = sol.icontrols["all"]
    # P = sol.parameters['all']
    P = sol.parameters
    steps = sol.ocp.nlp[0].ode_solver.steps
    ns_int = ns + 1

    forces = np.zeros((2, ns_int))
    for ii in range(ns_int):
        forces[[0, 1], ii] = np.squeeze(
            ocp.nlp[0].contact_forces_func(X[:, ii], U[:, ii], P).toarray())
    return forces


forces = Compute_all_contact_forces(all_ocp[0], soli[0])

fig, axs = plt.subplots(2, figsize=(19.2, 10.8))
X = np.linspace(sol[i].phase_time[0], sol[i].phase_time[1], sol[i].ns[0] + 1);

ii = 0
Y = forces[ii, :]
axs[ii].plot(X, Y, label=label[0])
axs[ii].plot(X[0:100:5], Y[0:100:5], 'o', ms=2)
axs[ii].set(xlabel='Time (s)', ylabel=ylabels[jj])
axs[ii].set_title(vv + str(ii))

# plt.tight_layout()
# plt.legend()
# plt.savefig(cur_path + vv)
