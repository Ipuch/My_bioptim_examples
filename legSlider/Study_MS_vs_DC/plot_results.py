import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from bioptim import *
from biorbd import *
import os
cur_path = os.getcwd()
files = ['JumperMS.bo', "JumperDC.bo"]
sol = []
soli = []
for i, v in enumerate(files):
    ocp, solution = OptimalControlProgram.load(cur_path +'/'+ v)
    sol.append(solution)

soli.append(sol[0].integrate(shooting_type=Shooting.MULTIPLE,
                             keep_intermediate_points=True,
                             ))  #, keepdims=False))
soli.append(sol[1].integrate(shooting_type=Shooting.MULTIPLE,
                             keep_intermediate_points=True,
                             continuous=False))

# sol[0].graphs()
# sol[1].graphs()

var = ['q', 'qdot', 'tau']
label = ['MS', 'DC']
ylabels = ['Angle', 'Angle Velocity', 'Torque (N)']
cc=['blue', 'red']
N = len(label)
# --------------------------------------------
# TAU
vv="tau"
fig, axs = plt.subplots(3, figsize=(19.2, 10.8))
for i, v in enumerate(files):
    for ii in range(3):  # nb ddl
         for p in range(2):
            Ntot = sol[i].ns[p] * 5 + 1
            X = np.linspace(sol[i].phase_time[0+p], sol[i].phase_time[1+p], Ntot);

            Y = np.zeros(sol[i].ns[p]*5 + 1)
            idx = list(range(sol[i].ns[p] * 5 + 1))
            Y[0:Ntot:5] = sol[i].controls[p][vv][ii, :]
            for kk in range(1, 5):
                Y[idx[kk:Ntot:5]] = sol[i].controls[p][vv][ii, :-1]

            axs[ii].plot(X, Y, label=label[i],color=cc[i])
            axs[ii].plot(X[0:Ntot:5], Y[0:Ntot:5],'o', ms=2,color=cc[i])
            axs[ii].set(xlabel='Time (s)', ylabel=ylabels[2])
            axs[ii].set_title(vv + str(ii))
            axs[ii].plot(sol[i].phase_time[1+p], 0, 'o', ms=5, color='black')
            plt.show()
plt.tight_layout()
plt.legend()

plt.savefig(cur_path + '/' + vv)
# --------------------------------------------
# q
vv = "q"
fig, axs = plt.subplots(3, figsize=(19.2, 10.8))
for i, v in enumerate(files):
    for ii in range(3):  # nb ddl
         for p in range(2):
            Ntot = sol[i].ns[p] * 5 + 1
            X = np.linspace(sol[i].phase_time[0+p], sol[i].phase_time[1+p], Ntot);

            Y = soli[i].states[p][vv][ii, :]

            axs[ii].plot(X, Y, label=label[i],color=cc[i])
            axs[ii].plot(X[0:Ntot:5], Y[0:Ntot:5],'o', ms=2,color=cc[i])
            axs[ii].set(xlabel='Time (s)', ylabel='Angle',)
            axs[ii].set_title(vv + str(ii))
            axs[ii].plot(sol[i].phase_time[1+p], 0, 'o', ms=5, color='black')
            plt.show()
plt.tight_layout()
plt.legend()
plt.savefig(cur_path + '/' + vv)
# --------------------------------------------
# qdot
vv = "qdot"
fig, axs = plt.subplots(3, figsize=(19.2, 10.8))
for i, v in enumerate(files):
    for ii in range(3):  # nb ddl
         for p in range(2):
            Ntot = sol[i].ns[p] * 5 + 1
            X = np.linspace(sol[i].phase_time[0+p], sol[i].phase_time[1+p], Ntot);

            Y = soli[i].states[p][vv][ii, :]

            axs[ii].plot(X, Y, label=label[i],color=cc[i])
            axs[ii].plot(X[0:Ntot:5], Y[0:Ntot:5],'o', ms=2,color=cc[i])
            axs[ii].set(xlabel='Time (s)', ylabel='Angle Velocity',)
            axs[ii].set_title(vv + str(ii))
            axs[ii].plot(sol[i].phase_time[1+p], 0, 'o', ms=5, color='black')
            plt.show()
plt.tight_layout()
plt.legend()
plt.savefig(cur_path + '/' + vv)

# --------------------------------------------
# qdot
vv = "qdot"
fig, axs = plt.subplots(3, figsize=(19.2, 10.8))
for i, v in enumerate(files):
    for ii in range(3):  # nb ddl
         for p in range(2):
            Ntot = sol[i].ns[p] * 5 + 1
            X = np.linspace(sol[i].phase_time[0+p], sol[i].phase_time[1+p], Ntot);

            Y = soli[i].states[p][vv][ii, :]

            axs[ii].plot(X, Y, label=label[i],color=cc[i])
            axs[ii].plot(X[0:Ntot:5], Y[0:Ntot:5],'o', ms=2,color=cc[i])
            axs[ii].set(xlabel='Time (s)', ylabel='Angle Velocity',)
            axs[ii].set_title(vv + str(ii))
            axs[ii].plot(sol[i].phase_time[1+p], 0, 'o', ms=5, color='black')
            plt.show()
plt.tight_layout()
plt.legend()
plt.savefig(cur_path + '/' + vv)


time_to_optimize = [sol[i].time_to_optimize for i in range(len(files))]

plt.figure()
plt.bar(x=list(range(N )), height=time_to_optimize)
plt.xlabel('method')
plt.ylabel('time (s)')
plt.show()
locs, labels = plt.xticks()  # Get the current locations and labels.
plt.xticks(np.arange(0, N , step=1))  # Set label locations.
plt.xticks(np.arange(N ), label, rotation=20)  # Set text labels.
plt.tight_layout()
plt.title('Time to converge for each Method')
plt.savefig(cur_path + '/optimization_time')

CostFunctionVal = [sol[i].cost.toarray() for i in range(len(files))]
CostFunctionVal = np.squeeze(np.concatenate(CostFunctionVal, axis=0))

plt.figure()
plt.bar(x=list(range(N)), height=CostFunctionVal)
plt.xlabel('method')
plt.show()
locs, labels = plt.xticks()  # Get the current locations and labels.
plt.xticks(np.arange(0, N, step=1))  # Set label locations.
plt.xticks(np.arange(N), label, rotation=20)  # Set text labels.
plt.tight_layout()
plt.title('Cost Function Values')
plt.savefig(cur_path + '/cost_function_val')


Iter = [sol[i].iterations for i in range(len(files))]

plt.figure()
plt.bar(x=list(range(N)), height=Iter)
plt.xlabel('method')
plt.show()
locs, labels = plt.xticks()  # Get the current locations and labels.
plt.xticks(np.arange(0, N, step=1))  # Set label locations.
plt.xticks(np.arange(N), label, rotation=20)  # Set text labels.
plt.tight_layout()
plt.title('Cost Function Values')
plt.savefig(cur_path + '/Iter')





