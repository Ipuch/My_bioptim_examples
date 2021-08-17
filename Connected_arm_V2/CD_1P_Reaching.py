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
    PlotType
)
# k = 10


def passive_moment(q: MX,  q_to_plot: list, stiffness: float = 10,
                   q0: float = 0) -> MX:
    """
    Create a used defined plot function with extra_parameters

    Parameters
    ----------
    q: MX
        The current states of the optimization
    q_to_plot: list
        The slice indices to plot
    stiffness
        float
    q0
        float
    Returns
    -------
    The value to plot
    """

    return k * (q[q_to_plot, :] - q0)


def compute_dynamics(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp, with_contact=True, stiffness: float = 10,
                   q0: float = 0) -> tuple:
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
    with_contact
    stiffness
        float
    q0
        float
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
    k = stiffness

    tau_p[4] = passive_moment(q, 4, k, q0)

    return q, qdot, tau+tau_p


def custom_dynamic(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp, with_contact=True, stiffness: float = 10,
                   q0: float = 0) -> tuple:
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
    with_contact
    stiffness
        float
    q0
        float
    Returns
    ----------
    MX.sym
        The derivative of the states
    """
    q, qdot, tau = compute_dynamics(states, controls, parameters, nlp, with_contact, stiffness, q0)

    dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
    # dqq = nlp.model.ForwardDynamics
    ddq = nlp.model.ForwardDynamicsConstraintsDirect(q, qdot, tau).to_mx()

    return dq, ddq


def custom_contact(states, controls, parameters, nlp, with_contact: bool = True, stiffness: float = None, q0: float = 0) -> tuple:
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
    with_contact
    stiffness
        float
    q0
        float
    Returns
    ----------
    MX.sym
        The derivative of the states
    """
    q, qdot, tau = compute_dynamics(states, controls, parameters, nlp, with_contact, stiffness, q0)

    # dqq = nlp.model.ForwardDynamics
    contact = nlp.model.ContactForcesFromForwardDynamicsConstraintsDirect(q, qdot, tau).to_mx()

    return contact


def dynamic_config(ocp: OptimalControlProgram, nlp: NonLinearProgram, with_contact=True, stiffness=10, q0=0):
    ConfigureProblem.configure_q(nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qdot(nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_tau(nlp, as_states=False, as_controls=True)
    ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamic, with_contact=with_contact,
                                                 stiffness=stiffness, q0=q0)

    if with_contact:
        ConfigureProblem.configure_contact_function(ocp, nlp, custom_contact)


def prepare_ocp(
        biorbd_model_path: str = "SliderXY_1Leg.bioMod",
        ode_solver: OdeSolver = OdeSolver.RK4(),
        k: float = 5,
        q0: float = 0,
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
    q0
        rad
    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = biorbd.Model(biorbd_model_path),

    n_shooting = 20,
    final_time = 0.5,

    tau_min, tau_max, tau_init = -2000, 2000, 0

    dynamics = Dynamics(dynamic_config, with_contact=True, dynamic_function=custom_dynamic, stiffness=k, q0=q0)

    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=0)
    # objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS, weight=100, first_marker="m0",
                            # second_marker="mg3")
    objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS, weight=100000, first_marker="ms1",
                            second_marker="mg2")

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
    # constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="ms1", second_marker="mg2")

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
    k = 20
    q0 = np.pi/2+0.01
    ocp = prepare_ocp("SliderXY_1Leg.bioMod", OdeSolver.RK4(), k, q0)

    ocp.add_plot("My New Extra Plot", lambda x, u, p: passive_moment(x, 4, k, q0), plot_type=PlotType.PLOT)

    ocp.print(to_console=True, to_graph=False)

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    # sol.animate()
    # sol.graphs()
    sol.print()

    # --- results --- #
    tau = sol.controls['tau']
    q = sol.states['q']
    ii = 0
    t0 = sol.phase_time[ii]
    t1 = sol.phase_time[ii + 1]
    ns = sol.ns[ii]

    import matplotlib.pyplot as plt

    plt.figure()

    tauLeg = ocp.nlp[0].plot['tau_controls'].legend
    plt.plot(np.linspace(t0, t0 + t1, ns + 1), tau.T, "-", label=tauLeg)
    plt.plot(np.linspace(t0, t0 + t1, ns + 1), k*(q[4]-q0), ".-", label="Passive_Torque")

    plt.legend()
    plt.grid()
    plt.title("Tau")
    plt.show()

    plt.figure()
    qLeg = ocp.nlp[0].plot['q_states'].legend
    plt.plot(np.linspace(t0, t0 + t1, ns + 1), q.T, "-", label=qLeg)

    plt.legend()
    plt.grid()
    plt.title("q")
    plt.show()