import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from bioptim import *
from biorbd import *
cur_path = '/examples_by_me/legSlider/Study_on_taudot_and_linear_continous/'
files = ['tau.bo' , "tau_LC.bo", 'dtau.bo' , "dtau_LC.bo"]
sol = []
soli = []
for i, v in enumerate(files):
    ocp, solution = OptimalControlProgram.load(cur_path + v)
    sol.append(solution)
    soli.append(solution.integrate(shooting_type=Shooting.MULTIPLE, keepdims=False))

var = ['q', 'qdot', 'tau']
label = ['Tau driven', 'Tau driven PiecewiseLinear', 'Taudot driven', 'Taudot driven PiecewiseLinear']
ylabels = ['Angle', 'Angle Velocity', 'Torque (N)']
for jj, vv in enumerate(var):
    fig, axs = plt.subplots(3,figsize=(19.2,10.8))
    #fig.suptitle(vv)
    for i, v in enumerate(files):

        for ii in range(3):  # nb ddl
            # plt.plot(np.linspace(solution.phase_time[0], solution.phase_time[1], solution.ns[0]+1), solution.states['q'][0, :])
            X = np.linspace(sol[i].phase_time[0], sol[i].phase_time[1], sol[i].ns[0] * 5 + 1);

            if vv in list(sol[i].controls.keys()):
                if label[i] == 'Tau driven' :
                    Y = np.zeros(sol[i].ns[0]*5 + 1)
                    idx = list(range(sol[i].ns[0] * 5 + 1))
                    Y[0:101:5] = sol[i].controls[vv][ii, :]

                    for kk in range(1, 5):
                        Y[idx[kk:101:5]] = sol[i].controls[vv][ii, :-1]

                elif label[i] == 'Tau driven PiecewiseLinear':
                    x = np.linspace(sol[i].phase_time[0], sol[i].phase_time[1], sol[i].ns[0] + 1)
                    f = interp1d(x, sol[i].controls[vv][ii, :])
                    Y = f(X)

            else:
                Y = soli[i].states[vv][ii, :]

            axs[ii].plot(X, Y, label=label[i])
            axs[ii].plot(X[0:100:5], Y[0:100:5],'o', ms=2)
            axs[ii].set(xlabel='Time (s)', ylabel=ylabels[jj])
            axs[ii].set_title(vv + str(ii))

    plt.tight_layout()
    plt.legend()

    plt.savefig(cur_path + vv)
# solution.ocp.v.ocp.nlp[0].plot['q']

time_to_optimize = [sol[i].time_to_optimize for i in range(len(files))]

label = ['Tau driven', 'Tau driven \n PiecewiseLinear', 'Taudot driven', 'Taudot driven \n PiecewiseLinear']
plt.figure()
plt.bar(x=list(range(4)), height=time_to_optimize)
plt.xlabel('method')
plt.ylabel('time (s)')
plt.show()
locs, labels = plt.xticks()  # Get the current locations and labels.
plt.xticks(np.arange(0, 4, step=1))  # Set label locations.
plt.xticks(np.arange(4), label, rotation=20)  # Set text labels.
plt.tight_layout()
plt.title('Time to converge for each Method')
plt.savefig(cur_path + 'optimization_time')

CostFunctionVal = [sol[i].cost.toarray() for i in range(len(files))]
CostFunctionVal = np.squeeze(np.concatenate(CostFunctionVal, axis=0))

label = ['Tau driven', 'Tau driven \n PiecewiseLinear', 'Taudot driven', 'Taudot driven \n PiecewiseLinear']
plt.figure()
plt.bar(x=list(range(4)), height=CostFunctionVal)
plt.xlabel('method')
plt.ylabel('Torque squared')
plt.show()
locs, labels = plt.xticks()  # Get the current locations and labels.
plt.xticks(np.arange(0, 4, step=1))  # Set label locations.
plt.xticks(np.arange(4), label, rotation=20)  # Set text labels.
plt.tight_layout()
plt.title('Cost Function Values')
plt.savefig(cur_path + 'cost_function_val')



