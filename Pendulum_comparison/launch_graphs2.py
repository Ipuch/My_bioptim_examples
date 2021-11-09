import pickle
import numpy as np
import pandas as pd
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

p = '/home/puchaud/Projets_Python/My_bioptim_examples/Pendulum_comparison/'
f = open(p + 'df.pckl', 'rb')
df = pickle.load(f)
f.close()

pal = sns.color_palette(palette="coolwarm", n_colors=3)
pal.reverse()
markers = {"RK4": "s", "RK8": "h", "CVODES": "o",
           "IRK": "P", "COLLOCATION_legendre_3": "X", "COLLOCATION_legendre_9": "*"}
integrator_list = list(markers.keys())
nps_list = np.unique(df['node per second'].to_numpy())
tol_list = np.unique(df['optimizer tolerance'].to_numpy())

# str_values = ['constraints']
str_values = ['objective function value']
# str_values = ['dynamic consistency']
# str_values = ['time']

# fig, ax = plt.subplots(1, len(str_values), figsize=(12, 8))
n = len(markers)
N = 3
width = 0.2
x = np.arange(n)
X = np.zeros((n, N))
for ii in range(N):
    X[:, ii] = x + (-N / 2 + 1 / 2 + ii) * width

for i in range(len(str_values)):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    for ii, tol in enumerate(tol_list):
        ddf = df[df["optimizer tolerance"] == tol]
        D = ddf.pivot(index='integrator', columns="node per second", values=str_values[i])
        D = D.reindex(["RK4", "RK8", "IRK", "CVODES", "COLLOCATION_legendre_3", "COLLOCATION_legendre_9"])
        print(D)
        # D.plot(kind='bar', color=pal[ii], ax=ax, edgecolor='black')
        X_temp = X + (-N / 2 + 1 / 2 + ii) * 0.02
        plt.plot(X_temp.T,
                 abs(D.values.T), 'o', markerfacecolor=pal[ii], ms=10, markeredgecolor="black", markeredgewidth=0.1,
                 alpha=0.9)

    my_labels = ["RK4", "RK8", "IRK", "CVODES",
                 "COLLOCATION\nlegendre 3", "COLLOCATION\nlegendre 9"]

    if len(str_values) == 1:
        AX = ax
    else:
        AX = ax[i]

    y_max = AX.get_ylim()

    a = [str(j) for j in nps_list]
    L = a * 6
    AX.set_xticks(X.ravel())
    AX.set_xticklabels(L)
    plt.setp(AX.get_xticklabels(), rotation=20)
    AX.set_ylabel(str_values)
    AX.set_yscale('log')
    AX.set_xlabel('Nodes')

    for ii in range(n):
        t = AX.text(x[ii], y_max[1], my_labels[ii], ha="center", va="bottom", rotation=0, size=9)
    AX.spines['top'].set_visible(False)
    AX.spines['right'].set_visible(False)

    AX.set_ylim(y_max)
    AX.vlines(np.array([0, 1, 2, 3, 4]) + 0.5, ymin=0, ymax=y_max[1], color='black', ls='--')

h_tol1 = AX.scatter([], [], c=np.array([pal[0]]), marker='o',
                    s=40, label=str(tol_list[0]))
h_tol2 = AX.scatter([], [], c=np.array([pal[1]]), marker='o',
                    s=40, label=str(tol_list[1]))
h_tol3 = AX.scatter([], [], c=np.array([pal[2]]), marker='o',
                    s=40, label=str(tol_list[2]))
plt.legend(handles=[h_tol1, h_tol2, h_tol3], labels=list(map(str, tol_list)), title='Tolerance', loc=(1.04, 0))

plt.tight_layout()
plt.show()
# plt.savefig(str_values[0] + '.jpg')
