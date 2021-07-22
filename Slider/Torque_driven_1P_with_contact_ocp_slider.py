"""
This example is a double slider that must reach each a height.
The first slider is constrained with a contact point.
The first slider cannot be actuated (tau=0)
The second slider as only the right to push (tau>0)
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
)

# Load the model  - for biorbd
biorbd_model = biorbd.Model("SliderContact.bioMod")

# Create a movement
n_frames = 100
q = np.zeros((biorbd_model.nbQ(), n_frames))
q[0,:]= np.linspace(-1, 1, n_frames)
q[1,:]= np.linspace(-1, 1, n_frames)
q.shape

# PROBLEM DE CONTROL OPTIMAL
# Dynamics definition
dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)
# State Boundaries conditions
x_bounds = QAndQDotBounds(biorbd_model)
# minimal and maximal bounds
# for all the degrees of freedom and velocities on three columns corresponding
# to the starting node, the intermediate nodes and the final node
x_bounds[0, 0] = 0
x_bounds[1, -1] = 1
x_bounds[1, 0] = 0  # velocity

# Control Boundary conditions
u_bounds = Bounds([0, -10], [0, 10])

# Objectif controls
objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

#Initial Guess
x_init = InitialGuess(np.zeros(4))
u_init = InitialGuess(np.zeros(2))

# Optimal Control Problem
ocp = OptimalControlProgram(
         biorbd_model,
        dynamics,
        n_shooting=500,
        phase_time=1,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
    )
#
sol = ocp.solve(show_online_optim=False)
#
sol.animate()
sol.graphs()

