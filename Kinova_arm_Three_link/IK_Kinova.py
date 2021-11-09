import biorbd as biorbd_eigen
from scipy import optimize
import numpy as np


def IK_Kinova_ThreeLink(model_path: str, q0: np.ndarray, targetd: np.ndarray, targetp: np.ndarray):
    """
    :param targetd:
    :param targetp:
    :param q0:
    :type model_path: object
    """
    m = biorbd_eigen.Model(model_path)
    bound_min = []
    bound_max = []
    for i in range(m.nbSegment()):
        seg = m.segment(i)
        for r in seg.QRanges():
            bound_min.append(r.min())
            bound_max.append(r.max())
    bounds = (bound_min, bound_max)

    def objective_function(x, *args, **kwargs):
        markers = m.markers(x)
        out1 = np.linalg.norm(markers[0].to_array() - targetd) ** 2
        out2 = np.linalg.norm(markers[-1].to_array() - targetp) ** 2
        return 2*out1 + out2

    pos = optimize.least_squares(objective_function, args=(m, targetd, targetp), x0=q0,
                                 bounds=bounds, verbose=2, method='trf',
                                 jac='3-point', ftol=2.22e-16, gtol=2.22e-16)
    # print(pos)
    print(f'Optimal q for the assistive arm at {targetd} is:\n{pos.x}\n'
          f'with cost function = {objective_function(pos.x)}')
    return pos.x