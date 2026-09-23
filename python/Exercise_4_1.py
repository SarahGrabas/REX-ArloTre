from robot import arlo
import numpy as np

ROBOT_TO_CAMERA = np.array([
    [1, 0, 0, 0],
    [0, 0, -1, 0.208],
    [0, 1, 0, -0.225],
    [0, 0, 0, 1]
])

CAMERA_TO_ROBOT = np.linalg.inv(ROBOT_TO_CAMERA)

BOX_CIRKEL_RADIUS = 0.25

arlo.start_camera()

ids, rvecs, tvecs = arlo.picDetectMarkersPose()

# Robot coords
# +X : Right
# +Y : Front
# +Z : Up 

# Camera coords
# +X : Right
# +Y : Down
# +Z : Front

map_array = [(
    1,
    CAMERA_TO_ROBOT @ np.array([x,y,z,1]),
    BOX_CIRKEL_RADIUS) 
    for id,(x,y,z) in zip(ids,tvecs)]

# Note:
# rvecs, tvecs
# [[ 3.11429633 -0.0338186   0.01586285]] [[0.01003483 0.04807003 0.37233776]]   ; looking right on 
# [[ 2.93820187 -0.02655488 -1.08660292]] [[-0.08554608  0.05370342  0.50980161]]   ; 45 degrees (2 faces visible)

