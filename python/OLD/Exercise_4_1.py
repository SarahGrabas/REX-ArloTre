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

print(rvecs, tvecs)

# Note:
# rvecs, tvecs
# [[ 3.11429633 -0.0338186   0.01586285]] [[0.01003483 0.04807003 0.37233776]]   ; boksen er lige frem, ikke roteret
# [[-0.01582608 -3.13767396  0.13222295]] [[-0.00227703  0.05177388  0.52987108]]   ; boksen er lige frem, på hovedet, ikke roteret
# [[ 2.93820187 -0.02655488 -1.08660292]] [[-0.08554608  0.05370342  0.50980161]]   ; boksen er lige frem, roteret 45 grader (2 sider synlige)
# [[ 2.1736008  -2.21441669  0.08033051]] [[0.00755181 0.095368   0.47405852]]   ; boksen er lige frem, væltet 90 grader

