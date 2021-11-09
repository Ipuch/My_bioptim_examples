"""
Not converging
"""

import biorbd_casadi as biorbd
import biorbd as biorbd_eigen
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
    CostType,
    PlotType,
)
import IK_Kinova
import CustomDynamics
import spring


def prepare_ocp(
        biorbd_model_path: str = "KINOVA_arm_reverse.bioMod",
        q0: np.ndarray = np.zeros((5, 1)),
        qfin: np.ndarray = np.zeros((5, 1)),
        springs: list = [],
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod
    q0:
    qfin:
    springs:
    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = biorbd.Model(biorbd_model_path),
    nbQ = biorbd_model[0].nbQ()

    n_shooting = 30,
    final_time = 0.5,

    tau_min, tau_max, tau_init = -20, 20, 0

    dynamics = Dynamics(CustomDynamics.dynamic_config, with_contact=True,
                        dynamic_function=CustomDynamics.custom_dynamic,
                        springs=springs)

    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=0)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=1, phase=0)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1, phase=0)

    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))

    x_bounds[0][nbQ:, 0] = 0  # 0 velocity at the beginning and the end to the phase
    x_bounds[0][nbQ:-3, -1] = 0

    u_bounds = BoundsList()
    u_bounds.add([tau_min] * nbQ,
                 [tau_max] * nbQ)

    u_bounds[0][2:, :] = 0

    x_init = InitialGuessList()
    x_init.add(q0.tolist() + [0] * nbQ)  # [0] * (nbQ+ nbQdot)

    u_init = InitialGuessList()
    u_init.add([tau_init] * nbQ)

    # Constraints
    constraints = ConstraintList()

    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="mg1", second_marker="md0")
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="mg2", second_marker="md0")

    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="grd_contact1",
                    second_marker="Contact_mk1")

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
        ode_solver=OdeSolver.RK4(),
        n_threads=8,

    )


if __name__ == "__main__":
    model = "KINOVA_arm_reverse.bioMod"

    m = biorbd_eigen.Model(model)
    X = m.markers()
    targetd = X[1].to_array()
    targetp_init = X[2].to_array()
    targetp_fin = X[3].to_array()

    q0 = np.array((targetp_init[0], targetp_init[1], 0, 1, 2))

    pos_init = IK_Kinova.IK_Kinova_ThreeLink(model, q0, targetd, targetp_init)
    pos_fin = IK_Kinova.IK_Kinova_ThreeLink(model, pos_init, targetd, targetp_fin)

    # Define passive torque parameters
    springs = [{}] * m.nbQ()
    springParam = dict(s=-1, k1=-2, k2=10, q0=0)
    springs[3] = spring.assignParam(springParam)
    springs[4] = spring.assignParam(springParam)
    # springParam[4] = dict(s=-1, k1=-2, k2=10, q0=0)

    ocp = prepare_ocp(model, pos_init, pos_fin, springs)
    # ocp.print(to_console=False, to_graph=True)

    # Custom plots
    ocp.add_plot_penalty(CostType.ALL)
    NP = "PassiveTorquePlot"
    ocp.add_plot(NP, lambda t, x, u, p: CustomDynamics.plot_passive_torque(x, u, p, ocp.nlp[0], springs, [3,4]),
                 plot_type=PlotType.INTEGRATED, axes_idx=[0, 1])
    # ocp.add_plot(NP, lambda t, x, u, p: CustomDynamics.plot_passive_torque(x, u, p, ocp.nlp[0], springs, 4),
    #              plot_type=PlotType.INTEGRATED, axes_idx=1)
    # ocp.add_plot(NP, lambda t, x, u, p: CustomDynamics.plot_passive_torque(x, u, p, ocp.nlp[0], springs, 3),
    # plot_type=PlotType.INTEGRATED, axes_idx=[1])
    # ocp.add_plot(NP, lambda t, x, u, p: plot_tot_torque(x, u, p, ocp.nlp[0], -0.1, 0)
    #              , plot_type=PlotType.INTEGRATED, axes_idx=[0], color='black')
    # ocp.add_plot(NP, lambda t, x, u, p: plot_tot_torque(x, u, p, ocp.nlp[0], 20, 1),
    #              plot_type=PlotType.INTEGRATED, axes_idx=[1], color='black')

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    sol.print()
    # ocp.save(sol, )
    sol.animate()
    sol.graphs()
