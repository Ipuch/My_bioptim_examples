"""
A optimal control program consisting in a triple pendulum starting downward and ending upward
while requiring the minimum of generalized forces. The solver is only allowed to move the pendulum sideways.
"""


import biorbd_casadi as biorbd
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

biorbd_model = biorbd.Model("Triple_Pendulum.bioMod")
#biorbd_model = biorbd.Model("pendulum.bioMod")
#bioviz.Viz("Triple_Pendulum.bioMod").exec()

# Dynamics definition
dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)
# State Boundaries conditions
x_bounds = QAndQDotBounds(biorbd_model)
#minimal and maximal bounds
# for all the degrees of freedom and velocities on three columns corresponding
# to the starting node, the intermediate nodes and the final node
x_bounds[:,[0,-1]]=0
x_bounds[0,-1]=3.14

# Control Boundary conditions
u_bounds = Bounds([-100, -50, -10], [100, 50, 10])

# Objectif controls
objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

#Initial Guess
x_init = InitialGuess([0, 0, 0, 0, 0, 0])
u_init = InitialGuess([0, 0, 0])

# Optimal Control Problem
ocp = OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting=20,
        phase_time=4,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        n_threads=8
    )

sol = ocp.solve(show_online_optim=False)

# sol.graphs()

import bioviz
sol.animate()