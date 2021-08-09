"""
Converge
"""

import biorbd_casadi as biorbd
import numpy as np
from casadi import MX, vertcat
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
    DynamicsFunctions,
    NonLinearProgram,
    ConfigureProblem,
)


def prepare_ocp(
        biorbd_model_path: str = "SliderXY_1Leg.bioMod",
        ode_solver: OdeSolver = OdeSolver.RK4(),
        k: float = 5,
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod
    ode_solver: OdeSolver
        The ode solve to use
    k
        N/rad
    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = biorbd.Model(biorbd_model_path),

    n_shooting = 20,
    final_time = 0.5,

    tau_min, tau_max, tau_init = -2000, 2000, 0

    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)

    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.1, phase=0)
    # objective_functions.add(ObjectiveFcn.Lagrange.SUPERIMPOSE_MARKERS, weight=100, first_marker="ms1",
    #                         second_marker="mg3")
    # objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS, weight=100, first_marker="ms1",
    #                         second_marker="mg2")

    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))

    x_bounds[0][:5, 0] = [0, 0, 0, 0, np.pi / 2]
    x_bounds[0][5:, 0] = 0
    x_bounds[0][5:, -1] = 0

    u_bounds = BoundsList()
    u_bounds.add([tau_min] * biorbd_model[0].nbGeneralizedTorque(),
                 [tau_max] * biorbd_model[0].nbGeneralizedTorque())

    u_bounds[0][3:, :] = 0

    x_init = InitialGuessList()
    x_init.add([0, 0, 0, 0, np.pi / 2] + [0] * biorbd_model[0].nbQdot())  # [0] * (nbQ+ nbQdot)

    u_init = InitialGuessList()
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())

    # Constraints
    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=-np.inf,
        max_bound=np.inf,
        node=Node.ALL,
    )

    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="ms1", second_marker="mg1")
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="ms1", second_marker="mg2")

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
    k = 10
    ocp = prepare_ocp("SliderXY_1Leg.bioMod", OdeSolver.RK4(), k)
    ocp.print(to_console=True, to_graph=False)

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=False)

    # --- Show results --- #
    sol.animate()
    # sol.graphs()

    import matplotlib.pyplot as plt

    plt.figure()
    ii = 0
    tau = sol.controls['tau']
    q = sol.states['q']
    t0 = sol.phase_time[ii]
    t1 = sol.phase_time[ii + 1]
    ns = sol.ns[ii]
    plt.plot(np.linspace(t0, t0 + t1, ns + 1), tau.T, ".-")  # legend=sol.controls.keys())

    # plt.legend()
    plt.grid()
    plt.title("Tau")
    plt.show()
