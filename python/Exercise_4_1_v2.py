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

ROBOT_TO_CAMERA = np.array([
    [1, 0, 0, 0],
    [0, 0, -1, 0.208],
    [0, 1, 0, -0.225],
    [0, 0, 0, 1]
])

CAMERA_TO_ROBOT = np.linalg.inv(ROBOT_TO_CAMERA)

BOX_SIZE=0.27

SIDE_TO_CENTER=[0,0, -BOX_SIZE/2] #negativ da vi går mod -Z aksen for boxen

max_box_size = max(MAX_BOX_SIZES.values())

arlo.start_camera()

ids, tvecs, rvecs, normals = arlo.picDetectMarkersPose()

zip(tvecs,rvecs)

centers=[]
for id, tvec, rvec in zip(ids,tvecs,rvecs):
    R, _ = cv2.Rodrigues(rvec)
    SIDE_TO_CENTER_camera=R @ SIDE_TO_CENTER #Vi roterer til kamera koordinater
    vektor_with_center= np.add(tvec,SIDE_TO_CENTER_camera)
    center= CAMERA_TO_ROBOT @ np.append(vektor_with_center, 1)

    centers.append(center[:3])
    
radiuses = [
    sqrt(MAX_BOX_SIZES.get(int(id), max_box_size) ** 2 / 2)
    for id in ids
]

coordinates_json = [
    {
        "id": int(id),
        "center": [float(i) for _, i in center],
        "radius": float(radius),
    }
    for id, center, radius in zip(ids,centers,radiuses)
]

with open("coordinates.json", "w") as f:
    json.dump(coordinates_json, f, indent=4)
    print(coordinates_json)

