"""
This example is a z-slider that must reach a height with an articulated leg of two segments.
The z-slider is not actuated.
The last segment of the leg is constrained with a contact point in the first phase and then released in the second phase
to reach the desired height minimizing torques.
Not converging.
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
    PhaseTransitionList,
    PhaseTransitionFcn
)


def prepare_ocp(
        biorbd_model_path: str = "Slider1Leg.bioMod",
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
    long_optim: bool
        If the solver should solve the precise optimization (500 shooting points) or the approximate (50 points)

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = biorbd.Model(biorbd_model_path) , biorbd.Model(biorbd_model_path)

    n_shooting = 25, 25
    final_time = 0.2, 0.4

    tau_min, tau_max, tau_init = -500, 500, 0

    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.1, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.1, phase=1)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=0.01, phase=0, min_bound=0.1, max_bound=0.2)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=0.01, phase=1, min_bound=0.3, max_bound=0.5)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.1, phase=0, derivative=True)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.1, phase=1, derivative=True)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_VELOCITY, weight=-1, axes=2, phase=0, quadratic=True)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_COM_POSITION, weight=-1, axes=2, phase=1, quadratic=True)

    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[1]))

    x_bounds[0][:3, 0] = [0, 3*np.pi / 8, -3*np.pi / 4]
    x_bounds[0][3:, 0] = 0

    x_bounds[0].min[0, -1] = 0
    x_bounds[0].max[0, -1] = 0.24  # from -0.15 to 0.25 (0.2469) otherwise it won't converge (singularity)

    x_bounds[1][0, -1] = 1
    x_bounds[1][3, -1] = 0

    u_bounds = BoundsList()
    u_bounds.add([tau_min] * biorbd_model[0].nbGeneralizedTorque(),
                 [tau_max] * biorbd_model[0].nbGeneralizedTorque())
    u_bounds.add([tau_min] * biorbd_model[1].nbGeneralizedTorque(),
                 [tau_max] * biorbd_model[1].nbGeneralizedTorque())

    u_bounds[0][0, :] = 0
    u_bounds[1][0, :] = 0

    x_init = InitialGuessList()
    x_init.add([0, 3*np.pi / 8, -3*np.pi / 4] + [0] * biorbd_model[0].nbQdot())  # [0] * (nbQ+ nbQdot)
    x_init.add([0, 0, 0] + [0] * biorbd_model[0].nbQdot())  # [0] * (nbQ+ nbQdot)

    u_init = InitialGuessList()
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
    u_init.add([tau_init] * biorbd_model[1].nbGeneralizedTorque())

    # Constraints
    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        phase=0,
        min_bound=0,
        max_bound=np.inf,
        node=Node.ALL_SHOOTING,
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
        n_threads=8,
    )


if __name__ == "__main__":
    """
    Defines a multiphase ocp and animate the results
    """
    ocp = prepare_ocp()

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=False)

    # --- Show results --- #
    sol.print()
    sol.animate()
    # sol.graphs()

    import matplotlib.pyplot as plt
    plt.figure()
    for ii in range(0, 2):
        tau = sol.controls[ii]['tau']
        t0 = sol.phase_time[ii]
        t1 = sol.phase_time[ii+1]
        ns = sol.ns[ii]
        plt.plot(np.linspace(t0, t0+t1, ns + 1), tau.T, ".-")

    # plt.legend()
    plt.grid()
    plt.title("Tau")
    plt.show()

    ocp.save(sol, "Study_MS_vs_DC/JumperMS.bo")