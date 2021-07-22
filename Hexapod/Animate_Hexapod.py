
import bioviz
import numpy as np

# Load the model  - for biorbd
# biorbd_model = biorbd.Model("Hexapod.bioMod")

# Load the model - for bioviz
biorbd_viz = bioviz.Viz("Hexapod.bioMod")
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
n_frames = 300
# Naive Hexapod Gait
q = np.ndarray((6, 24))
q[0, :] = np.array((0, 0, 0, 0, 0, 0, -45,  0, 0, 45,  0, 0,   0,  0, 0, -45,  0, 0, 45,  0, 0,  0,   0, 0)) * np.pi/180
q[1, :] = np.array((0, 0, 0, 0, 0, 0, -45, 30, 0, 45,  0, 0,   0, 30, 0, -45,  0, 0, 45, 30, 0,  0,   0, 0)) * np.pi/180
q[2, :] = np.array((0, 0, 0, 0, 0, 0, -15, 30, 0, 75,  0, 0, -30, 30, 0, -15,  0, 0, 75, 30, 0, -30,  0, 0)) * np.pi/180
q[3, :] = np.array((0, 0, 0, 0, 0, 0, -15,  0, 0, 75,  0, 0, -30,  0, 0, -15,  0, 0, 75,  0, 0, -30,  0, 0)) * np.pi/180
q[4, :] = np.array((0, 0, 0, 0, 0, 0, -15,  0, 0, 75, 30, 0, -30,  0, 0, -15, 30, 0, 75,  0, 0, -30, 30, 0)) * np.pi/180
q[5, :] = np.array((0, 0, 0, 0, 0, 0, -45,  0, 0, 45, 30, 0,   0,  0, 0, -45, 30, 0, 45,  0, 0,   0, 30, 0)) * np.pi/180

# Concatenate Array and create time series
Q = np.ndarray((24,250))
for ii, zz in zip(range(5), range(0, 300, 50)):
    Q[:, zz:(zz+50)] = np.array([np.linspace(i, j, int(n_frames/6)) for i, j in zip(q[ii, :], q[ii+1, :])])

# Animate the model
manually_animate = False
if manually_animate:
    i = 0
    while biorbd_viz.vtk_window.is_active:
        biorbd_viz.set_q(Q[:, i])
        i = (i+1) % n_frames
else:
    biorbd_viz.load_movement(Q)
    biorbd_viz.exec()

