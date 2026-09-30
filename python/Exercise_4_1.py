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

print(tuple(zip(ids, tvecs)))

max_box_size = max(MAX_BOX_SIZES.values())

# Box centers in camera coordinates 
centers = [
    tvec + normal * MAX_BOX_SIZES.get(int(id), max_box_size)/2
    for id, tvec, normal in zip(ids, tvecs, normals)
]

# Box centers in robot coordinates
centers = [
    (CAMERA_TO_ROBOT @ np.append(center, 1))[:3]
    for center in centers
]

radiuses = [
    sqrt(MAX_BOX_SIZES.get(int(id), max_box_size) ** 2 / 2)
    for id in ids
]


coordinates_json = [
    {
        "id": int(id),
        "center": [float(i) for i in center],
        "radius": float(radius),
    }
    for id, center, radius in zip(ids, centers, radiuses)
]

with open("coordinates.json", "w") as f:
    json.dump(coordinates_json, f, indent=4)
    print(coordinates_json)

