import biorbd
import bioviz
from bioptim import ObjectiveFcn, BiMapping
from scipy import optimize
import numpy as np

# model_name = "KINOVA_arm_deprecated.bioMod"
model_path = "/home/puchaud/Projets_Python/My_bioptim_examples/Kinova_arm/KINOVA_arm_reverse.bioMod"

m = biorbd.Model(model_path)
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
    out = np.linalg.norm(markers[0].to_array() - target)**2
    return out

target = np.zeros((1, 3))

pos = optimize.least_squares(objective_function, args=(m, target), x0=q0, bounds=bounds)
print(f"Optimal q for the assistive arm at {target} is:\n{pos.x}\n"
      f"with cost function = {objective_function(pos.x)}")

# Verification
q = np.tile(pos.x, (10, 1)).T
import bioviz
biorbd_viz = bioviz.Viz(model_path)
biorbd_viz.load_movement(q)
biorbd_viz.exec()

# TODO: without six dof & add plausible bounds
