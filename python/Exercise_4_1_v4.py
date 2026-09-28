from robot import arlo, cv2
import numpy as np

# ---------------------------------------------------------------------------
# Robot coords
# +X : Right
# +Y : Front
# +Z : Up 

# Camera coords
# +X : Right
# +Y : Down
# +Z : Front
# ---------------------------------------------------------------------------

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

def face_normal(rvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float))
    return R[:, 2]


map_array = [(
    1,
    CAMERA_TO_ROBOT @ np.array([x,y,z,1]),
    BOX_CIRKEL_RADIUS) 
    for id,(x,y,z) in zip(ids,tvecs)]



