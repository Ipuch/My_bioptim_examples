import matplotlib.pyplot as plt

import Pendulum2 as Pendulum
from bioptim import OdeSolver
import pickle
import numpy as np
import pandas as pd
from itertools import product
import biorbd_casadi as biorbd
import seaborn as sns

model_path = "models/triple_pendulum.bioMod"
biorbd_model = biorbd.Model(model_path)
nQ = biorbd_model.nbQ()

p = '/home/puchaud/Projets_Python/My_bioptim_examples/Triple_Pendulum_comparison/'
f = open(p + 'd.pckl', 'rb')
d = pickle.load(f)
f.close()

time_vector = d['time']
states_list = d['states']
controls_list = d['controls']
states_rk45 = d['states_rk45']
integrator_names = d['integrator_names']
ns_per_second = d['ns_per_second']
tol = d['tol']
n_integrator = d['n_integrator']
n_node = d['n_node']
n_tol = d['n_tol']
# n_integrator = 6
# n_node = 3

pal = sns.color_palette(palette="coolwarm", n_colors=3)
# pal.reverse()
markers = {"RK4": "s", "RK8": "h", "CVODES": "o",
           "IRK": "P", "COLLOCATION_legendre_3": "X", "COLLOCATION_legendre_9": "*"}
my_labels = ["RK4", "RK8", "IRK\nlegendre 3", "CVODES",
                 "COLLOCATION\nlegendre 3", "COLLOCATION\nlegendre 9"]
# pal[1] = tuple(np.array(list(pal[1])) - 0.5)

# j = 0  # tolerance
# nb_col = int(np.ceil(np.sqrt(nQ)))
# fig, ax = plt.subplots(nb_col, nb_col)
# for i in range(nQ):
#     ind = np.unravel_index([i], (2, 2))
#     for i_int in range(n_integrator):
#         for i_ns in range(n_node):
#             ax[ind[0][0], ind[1][0]].plot(time_vector[i_ns][0, :, i_int, j], states_list[i_ns][i, :, i_int, j],
#                                           label=integrator_names[i_int] + ' ' + str(tol[j]) + ' ' + str(
#                                               ns_per_second[i_ns]))
#     ax[ind[0][0], ind[1][0]].set_title('q' + str(i))
#     # plt.legend()
# plt.show()
# fig.suptitle('States', fontsize=16)
#
# nb_col = int(np.ceil(np.sqrt(nQ)))
# fig, ax = plt.subplots(nb_col, nb_col)
# for i in range(nQ):
#     ind = np.unravel_index([i], (2, 2))
#     for i_int in range(n_integrator):
#         for i_ns in range(n_node):
#             ax[ind[0][0], ind[1][0]].plot(time_vector[i_ns][0, :, i_int, j], states_list[i_ns][i + nQ, :, i_int, j],
#                                           label=integrator_names[i_int] + ' ' + str(tol[j]) + ' ' + str(
#                                               ns_per_second[i_ns]))
#     ax[ind[0][0], ind[1][0]].set_title('qdot' + str(i))
#     # plt.legend()
# plt.show()
# fig.suptitle('Qdot', fontsize=16)
#
# nb_col = int(np.ceil(np.sqrt(nQ)))
# fig, ax = plt.subplots(nb_col, nb_col)
# for i in range(nQ):
#     ind = np.unravel_index([i], (2, 2))
#     for i_int in range(n_integrator):
#         for i_ns in range(n_node):
#             ax[ind[0][0], ind[1][0]].plot(time_vector[i_ns][0, :, i_int, j], controls_list[i_ns][i, :, i_int, j],
#                                           label=integrator_names[i_int] + ' ' + str(tol[j]) + ' ' + str(
#                                               ns_per_second[i_ns]))
#     ax[ind[0][0], ind[1][0]].set_title('u' + str(i))
#     # plt.legend()
# plt.show()
# fig.suptitle('Controls', fontsize=16)

fig, ax = plt.subplots(n_integrator, n_tol, sharey='all', sharex='all',figsize=(12, 8))
i_ns = 0
for i in range(nQ):
    # ind = np.unravel_index([i], (1, nQ))
    for i_int in range(n_integrator):
        ax[i_int, 0].set_ylabel(my_labels[i_int])
        # ax[i_int, i_tol].set_title(str((tol[i_tol]))
        for i_tol in range(n_tol):
            diff = states_list[i_ns][i, :, i_int, i_tol] - states_rk45[i_ns][i, :, i_int, i_tol]
            # ax[ind[0][0], ind[1][0]].plot(time_vector[i_ns][0, :, i_int, j], diff,
            #                               label=integrator_names[i_int] + ' ' + str(tol[j]) + ' ' + str(
            #                                   ns_per_second[i_ns]))
            # ax[i].plot(time_vector[i_ns][0, :, i_int, j], abs(diff),
            #            label=integrator_names[i_int] + ' ' + str(tol[i_tol]) + ' ' + str(
            #                ns_per_second[i_ns]), color=pal[i_tol], marker=list(markers.values())[i_int])
            ax[i_int, i_tol].plot(time_vector[i_ns][0, :, i_int, i_tol], abs(diff),
                                  label=integrator_names[i_int] + ' ' + str(tol[i_tol]) + ' ' + str(
                                      ns_per_second[i_ns]), color=pal[i_tol])

            # ax[ind[0][0], ind[1][0]].set_title('q' + str(i))
            # ax[ind[0][0], ind[1][0]].set_yscale('log')
            ax[i_int, i_tol].set_title(str((tol[i_tol])))
            # ax[i_int, i_tol].set_yscale('log')

# plt.legend(loc=(1.04, 0))
# plt.tight_layout()
ax[0, 0].set_ylim(bottom=1e-8)
ax[0, 0].set_yscale('log')
fig.suptitle('Dynamic consistency q0 q1 q2', fontsize=16)
plt.savefig('Dynamic_consistency_q1q2q3' + '.jpg')
plt.show()


# TODO : un graph pour chaque q avec les trois valeurs de noeuds. Le nombre de noeuds, épaissir la courbe.
# TODO : meme graph pour les states et controles.
fig, ax = plt.subplots(n_integrator, n_tol, sharey='all', sharex='all')
i_ns = 0
for i in range(nQ):
    # ind = np.unravel_index([i], (1, nQ))
    for i_int in range(n_integrator):
        ax[i_int, 0].set_ylabel(my_labels[i_int])
        # ax[i_int, i_tol].set_title(str((tol[i_tol]))
        for i_tol in range(n_tol):
            diff = states_list[i_ns][i, :, i_int, i_tol] - states_rk45[i_ns][i, :, i_int, i_tol]
            # ax[ind[0][0], ind[1][0]].plot(time_vector[i_ns][0, :, i_int, j], diff,
            #                               label=integrator_names[i_int] + ' ' + str(tol[j]) + ' ' + str(
            #                                   ns_per_second[i_ns]))
            # ax[i].plot(time_vector[i_ns][0, :, i_int, j], abs(diff),
            #            label=integrator_names[i_int] + ' ' + str(tol[i_tol]) + ' ' + str(
            #                ns_per_second[i_ns]), color=pal[i_tol], marker=list(markers.values())[i_int])
            ax[i_int, i_tol].plot(time_vector[i_ns][0, :, i_int, i_tol], abs(diff),
                                  label=integrator_names[i_int] + ' ' + str(tol[i_tol]) + ' ' + str(
                                      ns_per_second[i_ns]), color=pal[i_tol], ls=None)   # markerstyle='o'

            # ax[ind[0][0], ind[1][0]].set_title('q' + str(i))
            # ax[ind[0][0], ind[1][0]].set_yscale('log')
            ax[i_int, i_tol].set_title(str((tol[i_tol])))
            # ax[i_int, i_tol].set_yscale('log')

# plt.legend(loc=(1.04, 0))
# plt.tight_layout()
# ax[0, 0].set_ylim(bottom=1e-8)
# ax[0, 0].set_yscale('log')
plt.show()
fig.suptitle('Dynamic consistency q0 q1 q2', fontsize=16)



fig, ax = plt.subplots(n_integrator, n_tol, sharey='all', sharex='all')
i_ns = 0
for i_int in range(n_integrator):
    ax[i_int, 0].set_ylabel(my_labels[i_int])
    # ax[i_int, i_tol].set_title(str((tol[i_tol]))
    for i_tol in range(n_tol):
        diff = np.sum(np.abs(states_list[i_ns][:, :, i_int, i_tol] - states_rk45[i_ns][:, :, i_int, i_tol]),axis=0)
        # ax[ind[0][0], ind[1][0]].plot(time_vector[i_ns][0, :, i_int, j], diff,
        #                               label=integrator_names[i_int] + ' ' + str(tol[j]) + ' ' + str(
        #                                   ns_per_second[i_ns]))
        # ax[i].plot(time_vector[i_ns][0, :, i_int, j], abs(diff),
        #            label=integrator_names[i_int] + ' ' + str(tol[i_tol]) + ' ' + str(
        #                ns_per_second[i_ns]), color=pal[i_tol], marker=list(markers.values())[i_int])
        ax[i_int, i_tol].plot(time_vector[i_ns][0, :, i_int, i_tol], diff,
                              label=integrator_names[i_int] + ' ' + str(tol[i_tol]) + ' ' + str(
                                  ns_per_second[i_ns]), color=pal[i_tol])

        # ax[ind[0][0], ind[1][0]].set_title('q' + str(i))
        # ax[ind[0][0], ind[1][0]].set_yscale('log')
        ax[i_int, i_tol].set_title(str((tol[i_tol])))
        # ax[i_int, i_tol].set_yscale('log')

# plt.legend(loc=(1.04, 0))
# plt.tight_layout()
# ax[0, 0].set_ylim(bottom=1e-8)
ax[0, 0].set_yscale('log')
plt.show()
fig.suptitle('Mean absolute dynamic consistency q0 q1 q2', fontsize=16)


for i in range(nQ):
    fig, ax = plt.subplots(n_integrator, n_tol, sharey='all', sharex='all')
    fig.suptitle('q' + str(i), fontsize=16)
    ax[0, 0].set_yscale('log')
    for i_ns in range(n_node):
        # ind = np.unravel_index([i], (1, nQ))
        for i_int in range(n_integrator):
            ax[i_int, 0].set_ylabel(my_labels[i_int])
            # ax[i_int, i_tol].set_title(str((tol[i_tol]))
            for i_tol in range(n_tol):
                diff = states_list[i_ns][i, :, i_int, i_tol] - states_rk45[i_ns][i, :, i_int, i_tol]
                # ax[ind[0][0], ind[1][0]].plot(time_vector[i_ns][0, :, i_int, j], diff,
                #                               label=integrator_names[i_int] + ' ' + str(tol[j]) + ' ' + str(
                #                                   ns_per_second[i_ns]))
                # ax[i].plot(time_vector[i_ns][0, :, i_int, j], abs(diff),
                #            label=integrator_names[i_int] + ' ' + str(tol[i_tol]) + ' ' + str(
                #                ns_per_second[i_ns]), color=pal[i_tol], marker=list(markers.values())[i_int])
                ax[i_int, i_tol].plot(time_vector[i_ns][0, :, i_int, i_tol], abs(diff),
                                      label=integrator_names[i_int] + '\n ' + str(tol[i_tol]) + '\n ' + str(
                                          ns_per_second[i_ns]), color=pal[i_ns], lw=i_ns*2)
                plt.show()
                # ax[ind[0][0], ind[1][0]].set_title('q' + str(i))
                # ax[ind[0][0], ind[1][0]].set_yscale('log')
                ax[i_int, i_tol].set_title(str((tol[i_tol])))
                # ax[i_int, i_tol].set_yscale('log')
plt.legend(loc=(1.04, 0))
# plt.tight_layout()
# ax[0, 0].set_ylim(bottom=1e-8)

plt.show()
# fig.suptitle('Controls', fontsize=16)