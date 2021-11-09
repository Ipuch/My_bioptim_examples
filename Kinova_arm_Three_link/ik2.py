import biorbd
import bioviz
import numpy as np
import IK_Kinova

# model_name = "KINOVA_arm_deprecated.bioMod"
model = "KINOVA_arm_reverse.bioMod"

m = biorbd.Model(model)
X = m.markers()
targetd = X[1].to_array()
targetp_init = X[2].to_array()
targetp_fin = X[3].to_array()

q0 = np.array((targetp_init[0], targetp_init[1], 0, 1, 2))

pos_init = IK_Kinova.IK_Kinova_ThreeLink(model, q0, targetd, targetp_init)
pos_fin = IK_Kinova.IK_Kinova_ThreeLink(model, pos_init, targetd, targetp_fin)

q = np.linspace(q0, pos_init, 20)
qq = np.linspace(pos_init, pos_fin, 20)
Q = np.concatenate((q, qq)).T
biorbd_viz = bioviz.Viz(model)
biorbd_viz.load_movement(Q)
biorbd_viz.exec()
