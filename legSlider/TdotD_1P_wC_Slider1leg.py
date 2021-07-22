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
    ControlType,
)


def prepare_ocp(
    biorbd_model_path: str = "Slider1Leg.bioMod", ode_solver: OdeSolver = OdeSolver.RK4()
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

    biorbd_model = (biorbd.Model(biorbd_model_path),)
    nq = biorbd_model[0].nbQ()

    n_shooting = (20,)
    final_time = (0.2,)

    tau_min, tau_max, tau_init = -2000, 2000, 0

    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DERIVATIVE_DRIVEN, with_contact=True)

    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.1, phase=0)

    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    x_bounds[0].concatenate(
        Bounds([tau_min] * biorbd_model[0].nbGeneralizedTorque(), [tau_max] * biorbd_model[0].nbGeneralizedTorque())
    )

    x_bounds[0][:3, 0] = [0, 3 * np.pi / 8, -3 * np.pi / 4]
    x_bounds[0][3:, 0] = 0
    x_bounds[0][0, -1] = 0.2469  # from -0.15 to 0.25 (0.2469) otherwise it won't converge

    x_bounds[0][2 * nq, :] = 0

    u_bounds = BoundsList()
    u_bounds.add(
        [tau_min * 10] * biorbd_model[0].nbGeneralizedTorque(), [tau_max * 10] * biorbd_model[0].nbGeneralizedTorque()
    )

    x_init = InitialGuessList()
    x_init.add(
        [0, 3 * np.pi / 8, -3 * np.pi / 4]
        + [0] * biorbd_model[0].nbQdot()
        + [tau_init] * biorbd_model[0].nbGeneralizedTorque()
    )  # [0] * (nbQ+ nbQdot)

    u_init = InitialGuessList()
    u_init.add([tau_init * 10] * biorbd_model[0].nbGeneralizedTorque())

    # Constraints
    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=0,
        max_bound=np.inf,
        node=Node.ALL,
        contact_index=0,  # z axis
    )

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
        # control_type=ControlType.LINEAR_CONTINUOUS,
        n_threads=1,
    )


if __name__ == "__main__":
    """
    Defines a multiphase ocp and animate the results
    """
    ocp = prepare_ocp()
    ocp.print()

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=False)

    # --- Show results --- #
    # sol.animate()
    sol.graphs(automatically_organize=False)

    ocp.save(sol, "/home/puchaud/Projets_Python/Pyomeca_github/bioptim/examples_by_me/legSlider/dtau_LC")
    # ocp, solution = OptimalControlProgram.load(file_path)
