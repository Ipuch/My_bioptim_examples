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
    ConfigureProblem,
)

# k = 10

def custom_dynamic(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp, with_contact=None, k=None) -> tuple:
    """
     Forward dynamics driven by joint torques with springs.

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
    q = DynamicsFunctions.get(nlp.states["q"], states)
    qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
    tau = DynamicsFunctions.get(nlp.controls["tau"], controls)

    nb_tau = len(nlp.controls["tau"].index)
    tau_p = MX.zeros(nb_tau)

    # Spring parameters
    # k = 10
    l0 = 0.1

    tau_p[3] = -k * (q[3] - l0)

    dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
    ddq = nlp.model.ForwardDynamics(q, qdot, tau + tau_p).to_mx()
    print(tau_p[3])

    return dq, ddq


def dynamic_config(ocp: OptimalControlProgram, nlp: NonLinearProgram, with_contact=True, k=None):
    ConfigureProblem.configure_q(nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qdot(nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_tau(nlp, as_states=False, as_controls=True)
    ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamic, with_contact=with_contact, k=k)



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

    n_shooting = 20,
    final_time = 0.1,

    tau_min, tau_max, tau_init = -2000, 2000, 0

    dynamics = Dynamics(dynamic_config, with_contact=True, dynamic_function=custom_dynamic, k=20)

    objective_functions = ObjectiveList()
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.1, phase=0)
    # objective_functions.add(ObjectiveFcn.Lagrange.SUPERIMPOSE_MARKERS, weight=100, first_marker="m0",
                            # second_marker="mg3")
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
    # constraints.add(
    #     ConstraintFcn.TRACK_CONTACT_FORCES,
    #     # min_bound=0,
    #     # max_bound=np.inf,
    #     node=Node.ALL,
    #     contact_index=[0, 1],
    # )
    # constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker="md", second_marker="mg1")
    # constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="md", second_marker="mg2")
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
