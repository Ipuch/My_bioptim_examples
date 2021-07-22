"""
This example is a z-slider that must reach a height with an articulated leg of two segments minimizing torques.
The last segment of the leg is constrained with a contact point.
The z-slider is not actuated.
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
)


def custom_dynamic(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp) -> MX:
    """
     Forward dynamics driven by joint torques with contact constraints.

    Parameters
    ----------
    states: MX.sym
        The state of the system
    controls: MX.sym
        The controls of the system
    parameters: MX.sym
        The parameters of the system
    nlp: NonLinearProgram
    The definition of the system

    Returns
    ----------
    MX.sym
        The derivative of the states
    """
    DynamicsFunctions.apply_parameters(parameters, nlp)
    q, qdot, tau = DynamicsFunctions.dispatch_q_qdot_tau_data(states, controls, nlp)

    nb_tau = nlp.tau.shape[0]
    tau_p = MX.zeros(nb_tau)
    tau_p[3] = -10 * (q[3] - 0.1)

    qddot = biorbd.Model.ForwardDynamicsConstraintsDirect(nlp.model, q, qdot, tau + tau_p).to_mx()

    print(tau_p[3])
    # qdot = nlp.model.computeQdot(q, qdot).to_mx()
    # qdot_reduced = nlp.mapping["q"].to_first.map(qdot)
    # qddot_reduced = nlp.mapping["qdot"].to_first.map(qddot)

    return qdot, qddot


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
    long_optim: bool
        If the solver should solve the precise optimization (500 shooting points) or the approximate (50 points)

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = biorbd.Model(biorbd_model_path),

    n_shooting = 100,
    final_time = 0.1,

    tau_min, tau_max, tau_init = -2000, 2000, 0

    dynamics = DynamicsList()
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN, with_contact=True, dynamic_function=custom_dynamic)

    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=0.1, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.SUPERIMPOSE_MARKERS, weight=100, first_marker="m0",
                            second_marker="mg3")
    # objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS, weight=100, first_marker="md",
    #                         second_marker="mg2")

    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))

    x_bounds[0][4:, 0] = 0
    x_bounds[0][4:, 2] = 0

    u_bounds = BoundsList()
    u_bounds.add([tau_min] * biorbd_model[0].nbGeneralizedTorque(),
                 [tau_max] * biorbd_model[0].nbGeneralizedTorque())

    u_bounds[0][2, :] = 0
    u_bounds[0][3, :] = 0

    x_init = InitialGuessList()
    x_init.add([0] * biorbd_model[0].nbQ() + [0] * biorbd_model[0].nbQdot())  # [0] * (nbQ+ nbQdot)
    # x_init.add([-1.3, 2.6, -2.15, 0.2] + [0] * biorbd_model[0].nbQdot())  # [0] * (nbQ+ nbQdot)

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
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="m0", second_marker="mg3")

    # constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="md", second_marker="mg1")
    # constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="md", second_marker="mg2")
    # constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.ALL, first_marker="m0", second_marker="mg3")

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
    sol.animate()
    sol.graphs()
