"""
This example is a z-slider that must reach a height with an articulated leg of two segments minimizing torques.
The last segment of the leg is constrained with a contact point.
The z-slider is not actuated.
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
    ConstraintFcn,
    ConstraintList,
)


def prepare_ocp(
        biorbd_model_path: str = "ConnectedArm.bioMod",
        ode_solver: OdeSolver = OdeSolver.RK4()
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod
    ode_solver: OdeSolver
        The ode solve to use
    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = biorbd.Model(biorbd_model_path),

    n_shooting = 100,
    final_time = 0.2,

    tau_min, tau_max, tau_init = -2000, 2000, 0

    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)

    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=0.1, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.SUPERIMPOSE_MARKERS,
                            first_marker="mg3", second_marker="m0", weight=10000, phase=0)
    # objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS,
    #                         first_marker="md", second_marker="mg1", weight=5000, phase=0, node=Node.START)

    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))

    x_bounds[0][4:, 0] = 0

    u_bounds = BoundsList()
    u_bounds.add([tau_min] * biorbd_model[0].nbGeneralizedTorque(),
                 [tau_max] * biorbd_model[0].nbGeneralizedTorque())

    u_bounds[0][2, :] = 0
    u_bounds[0][3, :] = 0

    x_init = InitialGuessList()
    x_init.add([-1.3, 2.6, -2.15, 0.2] + [0] * biorbd_model[0].nbQdot())  # [0] * (nbQ+ nbQdot)

    u_init = InitialGuessList()
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())

    # Constraints
    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        # min_bound=0,
        # max_bound=np.inf,
        node=Node.ALL,
        contact_index=[0, 1],
    )
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="md", second_marker="mg1")
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="md", second_marker="mg2")
    # constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="mg3")
    # constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="m0", second_marker="mg3")

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        ode_solver=ode_solver,
        n_threads=8,

    )


if __name__ == "__main__":
    """
    Defines a multiphase ocp and animate the results
    """
    ocp = prepare_ocp()
    ocp.print(to_console=False, to_graph=True)

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=False)

    # --- Show results --- #
    sol.print()
    sol.animate()
    sol.graphs()
