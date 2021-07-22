"""
This example is a z-slider that must reach a height with an articulated leg of two segments.
The z-slider is not actuated.
The last segment of the leg is constrained with a contact point in the first phase and then released in the second phase
to reach the desired height minimizing torques.
"""

import biorbd_casadi as biorbd
import numpy as np
from casadi import MX

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


def marker_velocity_is_null(pn: PenaltyNode, first_marker: str) -> MX:
    """
    The used-defined constraint function to impose the velocity of a marker to be zero.

    Parameters
    ----------
    pn: PenaltyNode
        The penalty node elements
    first_marker: str
        The index of the first marker in the bioMod
    Returns
    -------
    The value that should be constrained in the MX format
    """
    # Get the index of the markers
    marker_0_idx = biorbd.marker_index(pn.nlp.model, first_marker)
    print(marker_0_idx)
    # Store the casadi function. Using add_casadi_func allow to skip if the function already exists
    markersVel_func = pn.nlp.add_casadi_func("markersVelocity",
                                             pn.nlp.model.markerVelocity,
                                             pn.nlp.q, pn.nlp.qdot,
                                             marker_0_idx)

    # Get the marker positions and compute the difference
    nq = pn.nlp.shape["q"]
    nqdot = pn.nlp.shape["qdot"]
    q = pn.x[:nq]
    dq = pn.x[nq: nq + nqdot]
    markersVel = markersVel_func(q, dq)
    print(markersVel[:, marker_0_idx])
    return markersVel[:, marker_0_idx]


def last_contact_frame_null(pn: PenaltyNode, idx_forces: np.ndarray) -> MX:
    """
    The used-defined constraint function to impose the contact to be null add the end of the phase.

    Parameters
    ----------
    pn: PenaltyNode
        The penalty node elements
    idx_forces: array
        The index of the force to be set to zero
    Returns
    -------
    The value that should be constrained in the MX format
    """
    # print(pn.x)
    # print(pn.x)
    # print(pn.u)
    # print(pn[1].x)
    # print(pn[1].u)
    # print(pn[0].nlp)
    force = pn.nlp.contact_forces_func(pn.nlp.states.cx, pn.nlp.controls.cx, pn.nlp.parameters.cx)
    # force = BiorbdInterface.mx_to_cx("forces", pn.nlp.contact_forces_func, pn.nlp.states, pn.nlp.controls,
    #                                  pn.nlp.parameters)
    print(force)
    # print(pn.x[:])
    # print(pn.x[-1])
    return force[np.array(idx_forces)]


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

    biorbd_model = biorbd.Model(biorbd_model_path), biorbd.Model(biorbd_model_path)

    n_shooting = 40, 40
    final_time = 0.2, 0.4

    tau_min, tau_max, tau_init = -200, 200, 0

    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.1, phase=0)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=1, phase=0)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=0.1, phase=1)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", derivative=True, weight=1, phase=1)
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TIME, weight=0.01, phase=1)

    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[1]))

    x_bounds[0][:3, 0] = [0, 3 * np.pi / 8, -3 * np.pi / 4]
    x_bounds[0][3:, 0] = 0
    x_bounds[0].min[0, -1] = 0
    x_bounds[0].max[0, -1] = 0.25  # from -0.15 to 0.25 (0.2469) otherwise it won't converge

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
    x_init.add([0, 3 * np.pi / 8, -3 * np.pi / 4] + [0] * biorbd_model[0].nbQdot())  # [0] * (nbQ+ nbQdot)
    x_init.add([0, 0, 0] + [0] * biorbd_model[0].nbQdot())  # [0] * (nbQ+ nbQdot)

    u_init = InitialGuessList()
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())
    u_init.add([tau_init] * biorbd_model[1].nbGeneralizedTorque())

    # Constraints
    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=0,
        max_bound=np.inf,
        node=Node.ALL,
        contact_index=0,  # z axis
    )
    # constraints.add(marker_velocity_is_null,
    #                 node=Node.ALL,
    #                 first_marker='m0',
    #                 phase=0)
    # fonction de tracking en plus de la fonction propre
    # index (which tracked)
    # target (to be tracked)
    constraints.add(ConstraintFcn.TRACK_MARKERS_VELOCITY, node=Node.ALL, phase=0,
                    index=0)
    # constraints.add(ConstraintFcn.TRACK_CONTACT_FORCES, node=Node.END, phase=0,
    #                 index=0)
    constraints.add(last_contact_frame_null,
                    node=Node.PENULTIMATE,
                    idx_forces=[0, 1],
                    phase=0)

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
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    sol.print()
    sol.animate()
    sol.graphs()
    sol
