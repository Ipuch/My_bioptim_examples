
import bioviz
import numpy as np

# Load the model  - for biorbd
# biorbd_model = biorbd.Model("Slider.bioMod")
# biorbd_model = biorbd.Model("pendulum.bioMod")
# Load the model - for bioviz
biorbd_viz = bioviz.Viz("pendulum.bioMod")
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
q = np.zeros((1, n_frames))
q[0,:]= np.linspace(-1, 1, n_frames)
q.shape
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
