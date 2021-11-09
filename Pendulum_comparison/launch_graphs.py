import pickle
import numpy as np
import pandas as pd
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt

p = '/home/puchaud/Projets_Python/My_bioptim_examples/Pendulum_comparison/'
f = open(p + 'df.pckl', 'rb')
df = pickle.load(f)
f.close()


DataFrame.to_csv(df)

pal = sns.color_palette(palette="coolwarm", n_colors=3)
pal.reverse()
markers = {"RK4": "s", "RK8": "h", "CVODES": "o",
           "IRK": "P", "COLLOCATION_legendre_3": "X", "COLLOCATION_legendre_9": "*"}
integrator_list = list(markers.keys())
nps_list = np.unique(df['node per second'].to_numpy())
tol_list = np.unique(df['optimizer tolerance'].to_numpy())


################# TIME ##########################

def get_data_from_my_df(integrator, nd, tol):
    ddf = df[df["integrator"] == integrator]
    dddf = ddf[ddf["node per second"] == nd]
    ddddf = dddf[dddf["optimizer tolerance"] == tol]
    d = ddddf["time"].to_numpy()
    return d[0]


fig, ax = plt.subplots()
fig.set_figheight(7)
fig.set_figwidth(10)

n = len(integrator_list)
width = 0.08  # the width of the bars
for i_tol, tol in enumerate(tol_list):
    data = np.zeros((len(integrator_list), len(nps_list)))
    for i_int, integrator in enumerate(integrator_list):
        for i_nps, nps in enumerate(nps_list):
            data[i_int, i_nps] = get_data_from_my_df(integrator, nps, tol)

    x = np.arange(len(nps_list))  # the label locations

    for jj in range(n):
        ax.scatter(data[jj, :], x + (n / 2 - 1 / 2 - jj) * width, s=200, label=integrator_list[jj],
                   marker=list(markers.values())[jj], c=np.array([pal[i_tol]]), edgecolor='face', alpha=0.7)

h_tol1 = ax.scatter([], [], c=np.array([pal[0]]), marker='o',
                    s=40, label=str(tol_list[0]))
h_tol2 = ax.scatter([], [], c=np.array([pal[1]]), marker='o',
                    s=40, label=str(tol_list[1]))
h_tol3 = ax.scatter([], [], c=np.array([pal[2]]), marker='o',
                    s=40, label=str(tol_list[2]))

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Time')
ax.set_yticks(x)
ax.set_yticklabels(nps_list)
ax.set_ylabel('Shooting nodes')
leg = ax.legend(integrator_list, title='Integrators', bbox_to_anchor=(1.04, 1), borderaxespad=0, loc="upper left")
for ii in range(len(leg.legendHandles)):
    leg.legendHandles[ii].set_color('black')
ax.add_artist(leg)
plt.legend(handles=[h_tol1, h_tol2, h_tol3], labels=list(map(str, tol_list)), title='Tolerance', loc=(1.04, 0))

plt.subplots_adjust(right=0.6)
fig.tight_layout(rect=[0, 0, 0.75, 1])
# plt.savefig("output.png", bbox_inches="tight")

plt.show()



######

g = sns.relplot(x="objective function value",
                y="constraints",
                hue="optimizer tolerance",
                size='node per second',
                style="integrator",
                markers=markers,
                data=df,
                sizes=(50, 300),
                alpha=.7,
                palette=pal,
                height=8,
                aspect=8 / 8,
                kind='scatter',
                edgecolor="black"
                )
g.set(yscale='log')
g.set(xscale='log')

g = sns.relplot(x="time",
                y="iter",
                hue="optimizer tolerance",
                size='node per second',
                style="integrator",
                markers=markers,
                data=df,
                sizes=(50, 300),
                alpha=.7,
                palette=pal,
                height=8,
                aspect=8 / 8,
                kind='scatter',
                edgecolor="black"
                )

g = sns.relplot(x="dynamic consistency",
                y="constraints",
                hue="optimizer tolerance",
                size='node per second',
                style="integrator",
                markers=markers,
                data=df,
                sizes=(50, 300),
                alpha=0.7,
                palette=pal,
                height=8,
                aspect=8 / 8,
                kind='scatter',
                edgecolor="black"
                )
g.set(yscale='log')
g.set(xscale='log')

#
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
#     g.map(sns.scatterplot, "time", list_data[ii], alpha=.5)
#     g.add_legend()
#
# pal = dict(RK4="royalblue", RK8="navy", COLLOCATION_radau="darkorange", CVODES="firebrick")
#
# g = sns.FacetGrid(df, col="node per second", hue="integrator", palette=pal)
# g.set(yscale='log')
# g.set(xscale='log')
# g.map(sns.scatterplot, 'dynamic consistency', "objective function value", alpha=.5)
# g.add_legend()
#
#
# g = sns.relplot(x='dynamic consistency',
#             y="objective function value",
#             hue="integrator",
#             size='node per second',
#             data=df,
#             sizes=(50, 300),
#             alpha=.7,
#             palette='muted',
#             height=8,
#             aspect=8/8)
# g.set(yscale='log')
# g.set(xscale='log')
#
#
# f, axes = plt.subplots(1, 3)
#
# g = sns.relplot(x="optimizer tolerance",
#             y="objective function value",
#             hue="integrator",
#             size='node per second',
#             data=df,
#             sizes=(50, 300),
#             alpha=.7,
#             palette='muted',
#             height=8,
#             aspect=8/8,
#             marker="+")
# g.set(yscale='log')
# g.set(xscale='log')
#
# g = sns.relplot(x="optimizer tolerance",
#             y="dynamic consistency",
#             hue="integrator",
#             size='node per second',
#             data=df,
#             sizes=(50, 300),
#             alpha=.7,
#             palette='muted',
#             height=8,
#             aspect=8/8)
# g.set(yscale='log')
# g.set(xscale='log')
#
# g = sns.relplot(x="optimizer tolerance",
#             y="time",
#             hue="integrator",
#             size='node per second',
#             data=df,
#             sizes=(50, 300),
#             alpha=.7,
#             palette='muted',
#             height=8,
#             aspect=8/8)
# g.set(yscale='log')
# g.set(xscale='log')
#
# g = sns.relplot(x="optimizer tolerance",
#             y="objective function value",
#             hue="integrator",
#             size='node per second',
#             data=df,
#             sizes=(50, 300),
#             alpha=.7,
#             palette='muted',
#             height=8,
#             aspect=8/8)
# g.set(yscale='log')
# g.set(xscale='log')

# pal = dict(RK4="royalblue", RK8="navy", COLLOCATION_radau="darkorange", CVODES="firebrick")
