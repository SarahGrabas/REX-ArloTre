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

print(zip(ids, [CAMERA_TO_ROBOT @ np.array([x,y,z,1]) for id,(x,y,z) in tvecs], (BOX_CIRKEL_RADIUS for _ in ids)))



