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
# ids, rvecs, tvecs
# [[1]] [[[ 3.06523904 -0.05187814  0.12851865]]] [[[-0.08507675  0.04995853  1.02999685]]] 

