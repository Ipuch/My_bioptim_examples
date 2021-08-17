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
    PlotType,
    ParameterList,
    InterpolationType,
)
from typing import Any

def set_k(biorbd_model: biorbd.Model, value: MX):
    pass


def set_q0(biorbd_model: biorbd.Model, value: MX):
    pass


def passive_moment(q: MX,  q_to_plot: list, k: float = None,
                   q0: float = None) -> MX:
    """
    Create a used defined plot function with extra_parameters

    Parameters
    ----------
    q: MX
        The current states of the optimization
    q_to_plot: list
        The slice indices to plot
    k
        float
    q0
        float
    Returns
    -------
    The value to plot
    """

    return k * (q[q_to_plot, :] - q0)


def dispatch_q_qdot_tau(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp, with_contact=True, stiffness: float = None,
                   q0: float = None) -> tuple:
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


def custom_contact(states, controls, parameters, nlp, with_contact: bool = True) -> tuple:
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
    stiffness = parameters[0]
    q0 = parameters[1]

    q, qdot, tau = dispatch_q_qdot_tau(states, controls, parameters, nlp, with_contact, stiffness, q0)

    # dqq = nlp.model.ForwardDynamics
    contact = nlp.model.ContactForcesFromForwardDynamicsConstraintsDirect(q, qdot, tau).to_mx()

    return contact


def custom_dynamic(states: MX.sym, controls: MX.sym, parameters: MX.sym, nlp, with_contact=True) -> tuple:
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
    stiffness=parameters[0]
    q0 = parameters[1]

    q, qdot, tau = dispatch_q_qdot_tau(states, controls, parameters, nlp, with_contact, stiffness, q0)

    dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
    # dqq = nlp.model.ForwardDynamics
    ddq = nlp.model.ForwardDynamicsConstraintsDirect(q, qdot, tau).to_mx()

    return dq, ddq


def dynamic_config(ocp: OptimalControlProgram, nlp: NonLinearProgram, with_contact=True):
    ConfigureProblem.configure_q(nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_qdot(nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_tau(nlp, as_states=False, as_controls=True)
    ConfigureProblem.configure_dynamics_function(ocp, nlp, custom_dynamic, with_contact=with_contact)

    if with_contact:
        ConfigureProblem.configure_contact_function(ocp, nlp, custom_contact)


def prepare_ocp(
        biorbd_model_path: str = "SliderXY_1Leg.bioMod",
        ode_solver: OdeSolver = OdeSolver.RK4(),
        p: float = None,
        lb: float = None,
        ub: float = None
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod
    ode_solver: OdeSolver
        The ode solve to use
    p
    lb
    ub
    Returns
    -------
    The OptimalControlProgram ready to be solved
    """
    k_init = p[0]
    min_k = lb[0]
    max_k = ub[0]

    q0_init = p[1]
    min_q0 = lb[1]
    max_q0 = ub[1]

    biorbd_model = biorbd.Model(biorbd_model_path),

    n_shooting = 20,
    final_time = 0.5,

    tau_min, tau_max, tau_init = -2000, 2000, 0

    dynamics = Dynamics(dynamic_config, with_contact=True, dynamic_function=custom_dynamic)

    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=0)
    objective_functions.add(ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS, weight=10000, first_marker="ms1",
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

    parameters = ParameterList()

    bound_k = Bounds(min_k, max_k, interpolation=InterpolationType.CONSTANT)
    initial_k = InitialGuess(k_init)

    parameters.add(
        "k",  # The name of the parameter
        set_k,
        initial_k,  # The initial guess
        bound_k,  # The bounds
        size=1,  # The number of elements this particular parameter vector has
        scaling=np.array([1]),
    )

    bound_q0 = Bounds(min_q0, max_q0, interpolation=InterpolationType.CONSTANT)
    initial_q0 = InitialGuess(q0_init)

    parameters.add(
        "q0",  # The name of the parameter
        set_q0,
        initial_q0,  # The initial guess
        bound_q0,  # The bounds
        size=1,  # The number of elements this particular parameter vector has
        scaling=np.array([1]),
    )

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
        parameters=parameters,
        ode_solver=ode_solver,
        n_threads=8,
    )


if __name__ == "__main__":
    """
    Defines a multiphase ocp and animate the results
    """
    k = 10
    q0 = np.pi/4
    p = [k, q0]
    lb = [0, -np.pi]
    ub = [50, np.pi]
    ocp = prepare_ocp("SliderXY_1Leg.bioMod", OdeSolver.RK4(), p, lb, ub)

    ocp.add_plot("My New Extra Plot", lambda t, x, u, p: passive_moment(x, 4, k, q0), plot_type=PlotType.PLOT)

    ocp.print(to_console=True, to_graph=False)

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=False)

    # --- Show results --- #
    sol.print()
    # print(sol.parameters)
    # sol.animate()
    # sol.graphs()

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
    plt.plot(np.linspace(t0, t0 + t1, ns + 1), k*q[4], ".-", label="Passive_Torque")

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