
import bioviz
import numpy as np

model_name = "ConnectedArm.bioMod"
# Load the model - for bioviz
biorbd_viz = bioviz.Viz(model_name)
#while biorbd_viz.vtk_window.is_active:
#    # Do some stuff...
#    biorbd_viz.refresh_window()

# Combines the two loadings
#bioviz.Viz(loaded_model=biorbd_model).exec()

# avec les options
#bioviz.Viz(loaded_model=biorbd_model,
#    show_meshes=True,
#    show_global_center_of_mass=True,
#    show_segments_center_of_mass=True,
#    show_global_ref_frame=True, show_local_ref_frame=True,
#    show_markers=False,
#    show_muscles=False,
#    show_analyses_panel=False
#).exec()

# Create a movement
n_frames = 100
q = np.zeros((biorbd_viz.nQ, n_frames))
for ii in range(biorbd_viz.nQ):
    q[ii, :] = np.linspace(-1, 1, n_frames)

# Animate the mode
manually_animate = False
if manually_animate:
    i = 0
    while biorbd_viz.vtk_window.is_active:
        biorbd_viz.set_q(q[:, i])
        i = (i+1) % n_frames
else:
    biorbd_viz.load_movement(q)
    biorbd_viz.exec()
