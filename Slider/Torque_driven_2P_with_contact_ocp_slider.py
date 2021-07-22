"""
This example is a double slider that must reach each a height.
The first slider is constrained with a contact point in the first phase.
There is no more contact in the second phase.
The first slider cannot be actuated (tau=0)
The second slider as only the right to push in the first phase (tau>0)
and only the right to pull in the second phase to drag the first slider to the desired height (tau<0).
"""

import biorbd_casadi as biorbd
import numpy as np
from bioptim import (
    PenaltyNode,
    OptimalControlProgram,
    DynamicsFcn,
    Dynamics,
    DynamicsList,
    Bounds,
    BoundsList,
    QAndQDotBounds,
    InitialGuess,
    InitialGuessList,
    ObjectiveFcn,
    Objective,
    ObjectiveList,
    OdeSolver,
    Node,
)


# Load the model  - for biorbd
# biorbd_model = biorbd.Model("Slider.bioMod")

def minimize_difference(all_pn: PenaltyNode):
    return all_pn[0].nlp.controls.mx - all_pn[1].nlp.controls.mx # to correct it's not working


def prepare_ocp(
        biorbd_model_path: str = "SliderContact.bioMod",
        ode_solver: OdeSolver = OdeSolver.RK4(),
        long_optim: bool = False
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod
    ode_solver: OdeSolver
        The ode solve to use
    long_optim: bool
        If the solver should solve the precise optimization (500 shooting points) or the approximate (50 points)

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = (biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path))

    # Problem parameters
    if long_optim:
        n_shooting = (100, 100)
    else:
        n_shooting = (20, 20)
    final_time = (0.5, 0.25)

    tau_min, tau_max, tau_init = -100, 100, 0

    # Add objective functions for each phases
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, phase=1)
    objective_functions.add(
        minimize_difference,
        custom_type=ObjectiveFcn.Mayer,
        node=Node.TRANSITION,
        weight=100,
        phase=1,
        quadratic=True,
    )

    # Dynamics for each phases
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))

    # Phase 0
    x_bounds[0][0, 0] = 0
    x_bounds[0][1, 0] = 0.5
    x_bounds[0][2:, 0] = 0
    print(x_bounds[0].max)
    # Phase 1
    x_bounds[1][0, -1] = 0.75
    x_bounds[1][1, -1] = 1
    # Velocities at zeros at the end
    x_bounds[1][2:, -1] = 0

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))  # [0] * (nbQ+ nbQdot)
    x_init.add([0] * (biorbd_model[0].nbQ() + biorbd_model[0].nbQdot()))

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds.add([tau_min] * biorbd_model[0].nbGeneralizedTorque(), [tau_max] * biorbd_model[0].nbGeneralizedTorque())
    u_bounds.add([tau_min] * biorbd_model[0].nbGeneralizedTorque(), [tau_max] * biorbd_model[0].nbGeneralizedTorque())

    # Torque of first actuator null on first phase and second phase
    u_bounds[0][0, :] = 0
    u_bounds[1][0, :] = 0
    # second actuator can only push in the first phase
    # second actuator can only pull in the second phase
    # u_bounds[0][1, -1] = 0
    # u_bounds[1][1, 0] = 0

    u_init = InitialGuessList()
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        ode_solver=ode_solver,
        n_threads=8
    )


if __name__ == "__main__":
    """
    Defines a multiphase ocp and animate the results
    """

    ocp = prepare_ocp(long_optim=True)

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=False)

    # --- Show results --- #
    sol.animate()
    sol.graphs()
