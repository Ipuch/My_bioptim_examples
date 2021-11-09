"""
This example is a trivial box that must superimpose one of its corner to a marker at the beginning of the movement and
a the at different marker at the end of each phase. Moreover a constraint on the rotation is imposed on the cube.
Finally, an objective for the transition continuity on the control is added. Please note that the "last" control
of the previous phase is the last shooting node (and not the node arrival).
It is designed to show how one can define a multiphase optimal control program
"""
import numpy as np
from typing import Callable, Union
from casadi import MX, horzcat, DM
import biorbd_casadi as biorbd
from bioptim import (
    PenaltyNode,
    OptimalControlProgram,
    Dynamics,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    BoundsList,
    Bounds,
    QAndQDotBounds,
    InitialGuessList,
    OdeSolver,
    Node,
    ConfigureProblem,
    NonLinearProgram,
    DynamicsFunctions,
    Axis,
    PhaseTransitionList,
    PhaseTransitionFcn,
    PlotType,
    InterpolationType,
)


def prepare_ocp(
        biorbd_model_path_0: str = "ContactModel_1DoF.bioMod",
        ode_solver: OdeSolver = OdeSolver.RK8(),
) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path_0: str
        The path to the bioMod
    ode_solver: OdeSolver
        The ode solve to use

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = biorbd.Model(biorbd_model_path_0)

    # Problem parameters
    n_shooting = 100
    final_time = 1
    tau_min, tau_max, tau_init = -1000, 1000, 0

    # Add objective functions
    objective_functions = ObjectiveList()
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,key="tau",weight=1000)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE,key="q",weight=1000,node=Node.ALL)

    # Dynamics

    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    # Constraints
    constraints = ConstraintList()

    # Path constraint
    x_bounds = QAndQDotBounds(biorbd_model)

    # Initial conditions
    x_bounds[0, 0] = 1
    x_bounds[1, 0] = 0
    # x_bounds[0, -1] = 1
    # x_bounds[1, -1] = 0

    u_bounds = BoundsList()
    n_tau = biorbd_model.nbGeneralizedTorque()
    u_bounds.add([tau_min] * n_tau, [tau_max] * n_tau)
    u_bounds[0][0, :] = 0

    # Initial guess
    x_init = InitialGuessList()
    x_init.add([0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()))

    u_init = InitialGuessList()
    u_init.add([tau_init] * n_tau)

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        ode_solver=ode_solver,
        objective_functions=objective_functions,
        constraints=constraints,
        n_threads=8,
    )


def main():
    ocp = prepare_ocp()

    # --- Solve the program --- #
    sol = ocp.solve()

    ocp.print(to_console=False, to_graph=False)
    # --- Show results --- #
    sol.animate()


if __name__ == "__main__":
    main()
