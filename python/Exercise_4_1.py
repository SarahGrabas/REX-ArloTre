from robot import arlo
import numpy as np

ROBOT_TO_CAMERA = np.array([
    [1, 0, 0, 0],
    [0, 0, -1, 0*0.225],
    [0, 1, 0, 0.208],
    [0, 0, 0, 1]
])

CAMERA_TO_ROBOT = np.linalg.inv(ROBOT_TO_CAMERA)

BOX_SIZE = 0.27

def rotate(r, v):
    """Returns v rotated with r (r is the axis-angle rotation representation)"""
    v = np.asarray(v, dtype=float)
    r = np.asarray(r, dtype=float)

    theta = np.linalg.norm(r)

    if theta < 1e-12:
        return v.copy()

    axis = r / theta
    xyz = v[:3]

    rotated = (
        xyz * np.cos(theta)
        + np.cross(axis, xyz) * np.sin(theta)
        + axis * np.dot(axis, xyz) * (1 - np.cos(theta))
    )

    if len(v) == 4:
        return np.append(rotated, v[3])
    elif len(v) == 3:
        return rotated
    else:
        raise ValueError("v skal have 3 eller 4 komponenter")


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

print([CAMERA_TO_ROBOT @ (rotate(r, np.array([0,0,BOX_SIZE/2,0])) + np.array([x,y,z,1])) for r,(x,y,z) in zip(rvecs,tvecs)])

