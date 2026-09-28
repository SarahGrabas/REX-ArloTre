from robot import arlo, cv2
import numpy as np
from math import sqrt
import json

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

MAX_BOX_SIZES = {
    1  : 0.28,
    2  : 0.275,
    3  : 0.28,
    4  : 0.285,
    5  : 0.275,
    6  : 0.31,
    7  : 0.37,
    8  : 0.245,
    9  : 0.245,
    10 : 0.275,
    11 : 0.305,
}

arlo.start_camera()

ids, tvecs, rvecs, normals = arlo.picDetectMarkersPose()

# centers in camera coords
centers = [tvec - normal * MAX_BOX_SIZES.get(id, max(MAX_BOX_SIZES.values())) for id, tvec, normal in zip(ids, tvecs, normals)]
# centers in robot coords
centers = [CAMERA_TO_ROBOT @ np.array([x,y,z,1]) for x,y,z in tvecs]

# assumes box is perfectly square
radiuses = [sqrt(2 * MAX_BOX_SIZES.get(id, max(MAX_BOX_SIZES.values())) ** 2) / 2 for id in ids]


coordinates_json = [
    {
        "id":id,
        "center":list(center),
        "radius":radius,
    }
    for id, center, radius in zip(ids, centers, radiuses)
]

with open("coordinates.json", "w") as f:
    json.dump(coordinates_json, f, indent=4)

