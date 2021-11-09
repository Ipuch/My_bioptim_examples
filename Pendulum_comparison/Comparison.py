import matplotlib.pyplot as plt

import Pendulum
from bioptim import OdeSolver
import pickle
import numpy as np
import pandas as pd
from itertools import product
import seaborn as sns

# integrator_list = [OdeSolver.COLLOCATION(polynomial_degree=5, method='radau')]

integrator_list = [OdeSolver.RK4(),
                   OdeSolver.RK8(),
                   OdeSolver.CVODES(),
                   OdeSolver.IRK(),
                   OdeSolver.COLLOCATION(polynomial_degree=3, method='legendre'),
                   OdeSolver.COLLOCATION(polynomial_degree=9, method='legendre')]
integrator_names = ['RK4', 'RK8', 'CVODES', 'IRK', 'COLLOCATION_legendre_3', 'COLLOCATION_legendre_9']

nstep = 1
ns_per_second = [50, 150, 250]
tol = [1e-2, 1e-5, 1e-8]

# integrator_names = ['RK4', 'RK8', 'IRK_legendre4', 'COLLOCATION_radau', 'CVODES']

df = pd.DataFrame(list(product(integrator_names, ns_per_second, tol)),
                  columns=['integrator', 'node per second', 'optimizer tolerance'])

n_integrator = len(integrator_list)
n_node = len(ns_per_second)
n_tol = len(tol)

sol_list = [[[[] for i in range(n_tol)] for j in range(n_node)] for k in range(n_integrator)]

t = np.zeros((n_integrator * n_node * n_tol, 1))
iterations = np.zeros((n_integrator * n_node * n_tol, 1))
f_obj = np.zeros((n_integrator * n_node * n_tol, 1))
dynamic_consistency = np.zeros((n_integrator * n_node * n_tol, 5))
constraints = np.zeros((n_integrator * n_node * n_tol, 1))
cpt = 0
constraints = np.zeros((n_integrator * n_node * n_tol, 1))
states_list = n_node * [[]]
controls_list = n_node * [[]]
time_vector = n_node * [[]]
for i_ns in range(n_node):
    states_list[i_ns] = np.zeros((4, ns_per_second[i_ns] + 1, n_integrator, n_tol))
    controls_list[i_ns] = np.zeros((2, ns_per_second[i_ns] + 1, n_integrator, n_tol))
    time_vector[i_ns] = np.zeros((1, ns_per_second[i_ns] + 1, n_integrator, n_tol))

for i_int in range(n_integrator):
    for i_ns in range(n_node):
        for i_tol in range(n_tol):
            print("Solve with")
            print("########## Integrator ##########")
            print(integrator_list[i_int])
            print("########## node / second ##########")
            print(ns_per_second[i_ns])
            print("########## Tolerance on IPOPT ##########")
            print(tol[i_tol])
            ocp, sol = Pendulum.main(ode_solver=integrator_list[i_int],
                                     n_shooting_per_second=ns_per_second[i_ns],
                                     tol=tol[i_tol])
            t[cpt, 0] = sol.real_time_to_optimize
            iterations[cpt, 0] = sol.iterations
            f_obj[cpt, 0] = sol.cost
            dynamic_consistency[cpt, :] = Pendulum.compute_error_single_shooting(ocp, sol, 1)
            constraints[cpt, 0] = np.sqrt(np.mean(sol.constraints.toarray() ** 2))

            if ocp.nlp[0].ode_solver.is_direct_collocation:
                n = ocp.nlp[0].ode_solver.polynomial_degree + 1
                states_list[i_ns][:, :, i_int, i_tol] = sol.states['all'][:, ::n]
            else:
                states_list[i_ns][:, :, i_int, i_tol] = sol.states['all']

            controls_list[i_ns][:, :, i_int, i_tol] = sol.controls['all']
            time_vector[i_ns][:, :, i_int, i_tol] = np.linspace(0, sol.phase_time[0 + 1], sol.ns[0] + 1)

            d = {
                "integrator": integrator_names[i_int],
                "node per second": ns_per_second[i_ns],
                "optimizer tolerance": tol[i_tol],
                "time": time_vector[i_ns][:, :, i_int, i_tol],
                "controls": controls_list[i_ns][:, :, i_int, i_tol],
                "states": states_list[i_ns][:, :, i_int, i_tol],
            }

            print("Done with")
            print("########## Integrator ##########")
            print(integrator_list[i_int])
            print("########## node / second ##########")
            print(ns_per_second[i_ns])
            print("########## Tolerance on IPOPT ##########")
            print(tol[i_tol])
            cpt = cpt + 1

print(sol_list)

df['time'] = t
df['iter'] = iterations
df['objective function value'] = f_obj
df['translation dynamic consistency'] = dynamic_consistency[:, 0]
df['rotation dynamic consistency'] = dynamic_consistency[:, 1]
df['linear velocity dynamic consistency'] = dynamic_consistency[:, 2]
df['angular velocity dynamic consistency'] = dynamic_consistency[:, 3]
df['dynamic consistency'] = dynamic_consistency[:, 4]
df['constraints'] = constraints[:, 0]

# for i in range(4):
#     plt.figure()
#     j = 0
#     # for j in range(4):
#     for i_int in range(n_integrator):
#         for i_ns in range(n_node):
#             plt.plot(time_vector[i_ns][0, :, i_int, j], states_list[i_ns][i, :, i_int, j],
#                      label=integrator_names[i_int] + ' ' + str(tol[j]) + ' ' + str(ns_per_second[i_ns]))
#     plt.title('states' + str(i))
#     plt.legend()
#     plt.show()

f = open('df' + '.pckl', 'wb')
pickle.dump(df, f)
f.close()

f = open('df.pckl', 'rb')
obj = pickle.load(f)
f.close()

# sns.pairplot(df, hue="integrator")
#
# g = sns.FacetGrid(df, row="node per second", hue="integrator")
# g.map(sns.scatterplot, "time", "objective function value", alpha=.7)
# g.add_legend()
# g.set(yscale='log')
#
# g = sns.FacetGrid(df, row="node per second", col="optimizer tolerance", hue="integrator")
# g.map(sns.scatterplot, "time", "objective function value", alpha=.7)
# g.add_legend()
# g.set(yscale='log')
#
# list_data = ['translation dynamic consistency',
#              'rotation dynamic consistency',
#              'linear velocity dynamic consistency',
#              'angular velocity dynamic consistency',
#              'dynamic consistency',
#              'constraints']
# for ii in range(len(list_data)):
#     g = sns.FacetGrid(df, col="optimizer tolerance", hue="integrator")
#     g.set(yscale='log')
#     g.map(sns.scatterplot, "time", list_data[ii], alpha=.7)
#     g.add_legend()
