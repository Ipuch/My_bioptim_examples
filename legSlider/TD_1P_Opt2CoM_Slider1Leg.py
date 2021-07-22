"""
This example is a double slider that must reach a height. The first slider is constrained with a contact point.
in the first phase of the OCP and then it is released. The second slider as only the right to push in the first phase
 and only the right to pull in the second phase to drag the first slider to the desired height.
It is designed to show how one can define its own parameter objective functions if the provided ones are not
sufficient.
In this example, the location of both CoMs of the two leg segments can be optimized
"""

from typing import Any

import numpy as np
from casadi import MX
import biorbd_casadi as biorbd
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
    ParameterList,
    InterpolationType,
)


def set_CoM(biorbd_model: biorbd.Model, value: MX, extra_value: Any):
    """
    The pre dynamics function is called right before defining the dynamics of the system. If one wants to
    modify the dynamics (e.g. optimize the CoM of segment 3 in this case)

    Parameters
    ----------
    biorbd_model: biorbd.Model
        The model to modify by the parameters
    value: MX
        The CasADi variables to modify the model
    extra_value: Any
        Any parameters required by the user. The name(s) of the extra_value must match those used in parameter.add
    segment_idx: int
        Index of the segment concerned
    """
    segment_idx = extra_value["segment_idx"]
    print(f'{" Segment index ":*^30}')
    print(f"{segment_idx}")
    print(f'{" Segment index  ":*^30}')
    biorbd_model.segment(segment_idx[0]).characteristics().setCoM(value[0:3])
    biorbd_model.segment(segment_idx[1]).characteristics().setCoM(value[3:])


def prepare_ocp(
        biorbd_model_path: str = "Slider1Leg.bioMod",
        ode_solver: OdeSolver = OdeSolver.RK4(),
        optim_CoM=True,
        max_CoM=None,
        min_CoM=None,
        initial_guess_CoM=None,
        segment_idx=None,
        use_sx=False) -> OptimalControlProgram:
    """
    Prepare the ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the bioMod
    ode_solver: OdeSolver
        The ode solve to use
    optim_CoM: bool
        If the solver should solve the precise optimization (500 shooting points) or the approximate (50 points)
    min_CoM: numpy array
    max_CoM: numpy array
    use_sx: bool
    initial_guess_CoM: numpy array
    segment_idx: int
        If the program should be constructed using SX instead of MX (longer to create the CasADi graph, faster to solve)
    Returns
    -------
    The OptimalControlProgram ready to be solved

    """

    if max_CoM is None:
        max_CoM = [0, 0.0, -.09]
    if min_CoM is None:
        min_CoM = [0, 0.0, -.11]
    if initial_guess_CoM is None:
        initial_guess_CoM = [0, 0.0, -.1]
    if segment_idx is None:
        segment_idx = int(0)

    biorbd_model = (biorbd.Model(biorbd_model_path),
                    )

    n_shooting = 40, #40
    final_time = 0.2, #0.4

    tau_min, tau_max, tau_init = -200, 200, 0

    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN, with_contact=True)

    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1, phase=0)

    x_bounds = BoundsList()
    x_bounds.add(bounds=QAndQDotBounds(biorbd_model[0]))

    x_bounds[0][:3, 0] = [0, 3 * np.pi / 8, -3 * np.pi / 4]
    x_bounds[0][3:, 0] = 0

    x_bounds[0][0, -1] = 0.2

    u_bounds = BoundsList()
    u_bounds.add([tau_min] * biorbd_model[0].nbGeneralizedTorque(),
                 [tau_max] * biorbd_model[0].nbGeneralizedTorque())

    u_bounds[0][0, :] = 0

    x_init = InitialGuessList()
    x_init.add([0, 3 * np.pi / 8, -3 * np.pi / 4] + [0] * biorbd_model[0].nbQdot())  # [0] * (nbQ+ nbQdot)

    u_init = InitialGuessList()
    u_init.add([tau_init] * biorbd_model[0].nbGeneralizedTorque())

    # Constraints
    constraints = ConstraintList()
    constraints.add(
        ConstraintFcn.TRACK_CONTACT_FORCES,
        min_bound=0,
        max_bound=np.inf,
        node=Node.ALL,
        contact_index=0,  # z axis
    )
    # Define the parameter to optimize
    parameters = ParameterList()

    if optim_CoM:
        nCoM = len(segment_idx)
        bound_CoM = Bounds(min_CoM * nCoM,
                           max_CoM * nCoM,
                           interpolation=InterpolationType.CONSTANT)
        initial_CoM = InitialGuess(initial_guess_CoM * nCoM)
        parameters.add(
            "CoMs",  # The name of the parameter
            set_CoM,  # The function that modifies the biorbd model
            initial_CoM,  # The initial guess
            bound_CoM,  # The bounds
            size=3*nCoM,  # The number of elements this particular parameter vector has
            scaling=np.array([1, 1, 1]*nCoM),
            extra_value=dict(value=1, segment_idx=segment_idx)  # You can define as many extra arguments as you want
        )

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        parameters=parameters,
        ode_solver=ode_solver,
        # use_sx=use_sx,
        n_threads=8,
    )


if __name__ == "__main__":
    """
    Solve and print the optimized value for the gravity and animate the solution
    """
    opti_CoM = True
    ocp = prepare_ocp(
        biorbd_model_path="Slider1Leg.bioMod",
        optim_CoM=opti_CoM,
        # Initial guess influence the final optimized value.
        max_CoM=[0, 1, .8],
        min_CoM=[0, -1, -.8],
        initial_guess_CoM=[0, 0, +.8],  # try .79, -.79, 0, 0.1
        segment_idx=[1, 2],
    )

    # --- Solve the program --- #
    sol = ocp.solve()

    # --- Get the results --- #
    if opti_CoM:
        CoM = sol.parameters["CoMs"]
        print(f'{" Results ":*^30}')
        print(f"Optimized CoM: {CoM}")
        print(f'{" Results ":*^30}')
    # initial guesses      .79,       -.79,         0, 0.1,  -.1    , -215
    # Res for z directions 0.0945419, - 0.945419,   0, 0.011, -0.011, -0.1
    # --- Show results --- #
    sol.animate(n_frames=200)
