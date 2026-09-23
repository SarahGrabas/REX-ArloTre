from robot import arlo
import numpy as np

ROBOT_TO_CAMERA = np.array([
    [1, 0, 0, 0],
    [0, 0, -1, 0.208],
    [0, 1, 0, -0.225],
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

#print([CAMERA_TO_ROBOT @ np.array([x,y,z,1]) for r,(x,y,z) in zip(rvecs,tvecs)])

for r,(x,y,z) in zip(rvecs,tvecs):
    marker_side_center= np.array([x,y,z])
    side_to_center_of_box = np.array([0,0,BOX_SIZE/2])
    
    side_camera_coords =rotate(r,side_to_center_of_box)
    box_center_camera = marker_side_center + side_camera_coords
    
    #In robot coords
    robot_coords=CAMERA_TO_ROBOT @ np.append(box_center_camera,1)
    print(robot_coords)
    
    print("Marker:", marker_side_center)
    print("Offset:", side_camera_coords)
    print("Box center camera:", box_center_camera)
    print("Box center robot:", robot_coords)

#print([CAMERA_TO_ROBOT @ (rotate(r, BOX_SIZE*np.array([x,y,z,0])/(2*np.linalg.norm([x,y,z,0]))) + np.array([x,y,z,1])) for r,(x,y,z) in zip(rvecs,tvecs)])

