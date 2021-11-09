"""
"""
import biorbd_casadi as biorbd
from bioptim import InitialGuess, Solution, Shooting, InterpolationType, OdeSolver, Dynamics, OptimalControlProgram, \
    DynamicsFcn
import numpy as np

import matplotlib.pyplot as plt


def prepare_ocp(
        biorbd_model_path: str,
        n_shooting: int,
        final_time: float,
        ode_solver: OdeSolver,
) -> OptimalControlProgram:
    """
    Prepare the program

    Parameters
    ----------
    biorbd_model_path: str
        The path of the biorbd model
    final_time: float
        The time at the final node
    n_shooting: int
        The number of shooting points
    ode_solver: OdeSolver
        The type of ode solver used

    Returns
    -------
    The ocp ready to be solved
    """

    biorbd_model = biorbd.Model(biorbd_model_path)

    # Add objective functions

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    # Initial guess
    x_init = InitialGuess([0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()))

    # Define control path constraint
    tau_min, tau_max, tau_init = -1000, 1000, 0

    u_init = InitialGuess([tau_init] * biorbd_model.nbGeneralizedTorque())

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        ode_solver=ode_solver,
    )


def main():
    t_fin = 3
    n_shooting = 2000

    # Simulation the Initial Guess
    # Interpolation: Constant
    X = InitialGuess([2, 0.5, 0, 0, 0, 10])
    U = InitialGuess([0, 0, 0])

    solver = OdeSolver.RK8()

    ocp = prepare_ocp("ContactModel_3DoF.bioMod", n_shooting, t_fin, ode_solver=solver)
    # ocp = prepare_ocp("ContactModel_3DoF.bioMod", n_shooting, t_fin, ode_solver=solver)

    # --- Solve the program --- #

    sol_from_initial_guess = Solution(ocp, [X, U])
    s = sol_from_initial_guess.integrate(shooting_type=Shooting.SINGLE, continuous=True)
    # sol_from_initial_guess.graphs(shooting_type=Shooting.SINGLE)
    s.animate(n_frames=200)
    ocp.print(to_console=True, to_graph=False)


if __name__ == "__main__":
    main()
