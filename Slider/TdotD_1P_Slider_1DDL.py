"""
This example is a trivial slider that must reach a height. y gravity
"""

import biorbd_casadi as biorbd
import numpy as np
from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    Dynamics,
    Bounds,
    QAndQDotBounds,
    InitialGuess,
    ObjectiveFcn,
    Objective,
    BoundsList,
)

# Load the model  - for biorbd
biorbd_model = biorbd.Model("Slider.bioMod")
nq = biorbd_model.nbQ()

# Create a movement
n_frames = 100
q = np.zeros((1, n_frames))
q[0, :] = np.linspace(-1, 1, n_frames)
# q.shape

# PROBLEM DE CONTROL OPTIMAL
# Dynamics definition
dynamics = Dynamics(DynamicsFcn.TORQUE_DERIVATIVE_DRIVEN)

# State Boundaries conditions
x_bounds = BoundsList()
x_bounds.add(bounds=QAndQDotBounds(biorbd_model))

# add torque bounds on the states
tau_min, tau_max, tau_init = -10, 10, 0
for dof in range(nq):
    x_bounds[0].concatenate(Bounds([tau_min], [tau_max]))

# minimal and maximal bounds
# for all the degrees of freedom and velocities on three columns corresponding
# to the starting node, the intermediate nodes and the final node
x_bounds[0][0, 0] = 0
x_bounds[0][0, -1] = 1

# Control Boundary conditions on torque derivatives
u_bounds = Bounds([tau_min * 10], [tau_max * 10])

# Objective function
objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="taudot")

# Initial Guess
x_init = InitialGuess(np.zeros(3))
u_init = InitialGuess(np.zeros(1))


# Optimal Control Problem
ocp = OptimalControlProgram(
    biorbd_model,
    dynamics,
    n_shooting=50,
    phase_time=4,
    x_init=x_init,
    u_init=u_init,
    x_bounds=x_bounds,
    u_bounds=u_bounds,
    objective_functions=objective_functions,
)
#
sol = ocp.solve(show_online_optim=False)
sol.graphs(automatically_organize=False)

import bioviz
sol.animate()
