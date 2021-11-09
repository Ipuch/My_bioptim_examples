"""
A very simple yet meaningful optimal control program consisting in a pendulum starting downward and ending upward
while requiring the minimum of generalized forces. The solver is only allowed to move the pendulum sideways.

This simple example is a good place to start investigating bioptim as it describes the most common dynamics out there
(the joint torque driven), it defines an objective function and some boundaries and initial guesses

During the optimization process, the graphs are updated real-time (even though it is a bit too fast and short to really
appreciate it). Finally, once it finished optimizing, it animates the model using the optimal solution
"""
import numpy as np
import biorbd_casadi as biorbd
from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    Dynamics,
    Bounds,
    QAndQDotBounds,
    InitialGuess,
    ObjectiveFcn,
    Objective,
    OdeSolver,
    CostType,
    Shooting,
    SolverOptionsIpopt,
)


def prepare_ocp(
        biorbd_model_path: str,
        final_time: float,
        ode_solver: OdeSolver = OdeSolver.RK4(),
        n_shooting_per_second: int = 30,
        use_sx: bool = True,
        n_threads: int = 1,
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the biorbd model
    final_time: float
        The time in second required to perform the task
    n_shooting_per_second: int
        The number of shooting points to define int the direct multiple shooting program by second
    ode_solver: OdeSolver = OdeSolver.RK4()
        Which type of OdeSolver to use
    use_sx: bool
        If the SX variable should be used instead of MX (can be extensive on RAM)
    n_threads: int
        The number of threads to use in the paralleling (1 = no parallel computing)

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """
    n_shooting = n_shooting_per_second * final_time

    biorbd_model = biorbd.Model(biorbd_model_path)

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    # Path constraint
    x_bounds = QAndQDotBounds(biorbd_model)
    x_bounds[:, [0, -1]] = 0
    x_bounds[1, -1] = 3.14
    x_bounds[1, [-2,-1]] = 3.14

    # Initial guess
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    x_init = InitialGuess([0] * (n_q + n_qdot))

    # Define control path constraint
    n_tau = biorbd_model.nbGeneralizedTorque()
    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = Bounds([tau_min] * n_tau, [tau_max] * n_tau)
    u_bounds[n_tau - 1, :] = 0

    u_init = InitialGuess([tau_init] * n_tau)

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        ode_solver=ode_solver,
        use_sx=False,
        n_threads=n_threads,
    )


def main(n_shooting_per_second: int,
         ode_solver: OdeSolver,
         tol: float):
    """
    If pendulum is run as a script, it will perform the optimization and animates it
    """

    # --- Prepare the ocp --- #
    ocp = prepare_ocp(biorbd_model_path="models/double_pendulum.bioMod", final_time=1,
                      n_shooting_per_second=n_shooting_per_second, ode_solver=ode_solver)

    # Custom plots
    # ocp.add_plot_penalty(CostType.ALL)

    # --- Print ocp structure --- #
    ocp.print(to_console=False, to_graph=False)

    # --- Solve the ocp --- #
    options = SolverOptionsIpopt()
    options.set_convergence_tolerance(tol)
    options.set_constraint_tolerance(tol)
    options.set_maximum_iterations(2000)
    options.limited_memory_max_history = 50
    options.linear_solver = "mumps"

    # --- Show the results in a bioviz animation --- #
    sol = ocp.solve(options)
    sol.print()

    return ocp, sol


def compute_error_single_shooting(ocp, sol, duration):
    if ocp.nlp[0].tf < duration:
        raise ValueError(
            f"Single shooting integration duration must be smaller than ocp duration :{ocp.nlp[0].tf} s"
        )
    sol_int = sol.integrate(shooting_type=Shooting.SINGLE_CONTINUOUS, use_scipy_integrator=True,
                            keep_intermediate_points=True)
    # print(ocp.nlp[0].ode_solver.polynomial_degree + 1)
    print(sol.states['q'].shape, sol_int.states['q'].shape)
    if sol.states['q'].shape[1] != 51:
        print('hello')
    ss_err_qdot_trans = np.sqrt(np.mean((sol_int.states["qdot"][0, -1] - sol.states["qdot"][0, -1]) ** 2))
    ss_err_qdot_rot = np.sqrt(np.mean((sol_int.states["qdot"][-1, -1] - sol.states["qdot"][-1, -1]) ** 2))
    ss_err_q_trans = np.sqrt(np.mean((sol_int.states["q"][0, -1] - sol.states["q"][0, -1]) ** 2))
    ss_err_q_rot = np.sqrt(np.mean((sol_int.states["q"][-1, -1] - sol.states["q"][-1, -1]) ** 2))

    ss_err_all = np.sqrt(np.mean((sol_int.states["q"][:, -1] - sol.states["q"][:, -1]) ** 2))

    print(f"Single shooting error for translation : {ss_err_q_trans / 1000} mm")
    print(f"Single shooting error for rotation : {ss_err_q_rot * 180 / np.pi} degrees")
    print(f"Single shooting error for translation velocity: {ss_err_qdot_trans / 1000} mm/s")
    print(f"Single shooting error for rotation velocity: {ss_err_qdot_rot * 180 / np.pi} degrees/s")

    return np.array([ss_err_q_trans, ss_err_q_rot, ss_err_qdot_trans, ss_err_q_rot, ss_err_all])


