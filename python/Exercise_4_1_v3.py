from robot import arlo
import numpy as np
import cv2 # type: ignore


# ---------------------------------------------------------------------------
# Coordinate transforms
#
# Robot:
#   +X = right
#   +Y = front
#   +Z = up
#
# Camera:
#   +X = right
#   +Y = down
#   +Z = front
# ---------------------------------------------------------------------------

ROBOT_TO_CAMERA = np.array([
    [1, 0,  0,     0],
    [0, 0, -1,  0.208],
    [0, 1,  0, -0.225],
    [0, 0,  0,     1],
], dtype=float)

CAMERA_TO_ROBOT = np.linalg.inv(ROBOT_TO_CAMERA)


# ---------------------------------------------------------------------------
# Box definitions
#
# Each box is defined in the MARKER coordinate system:
#
#   width  = size along marker-local X
#   height = size along marker-local Y
#   depth  = size going INTO the box from the marked face
# ---------------------------------------------------------------------------

BOX_WIDTHS = { # width is here the longest box edge, i.e. width = max(side_len1, side_len2)
    1: 0.22,
    # 2: 0.25,
}


# ---------------------------------------------------------------------------
# Convert an OpenCV rvec to a rotation matrix
#
# OpenCV's rvec describes:
#
#       marker coordinates -> camera coordinates
# ---------------------------------------------------------------------------

def rvec_to_matrix(rvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float))
    return R


# ---------------------------------------------------------------------------
# Calculate one box from one marker
# ---------------------------------------------------------------------------

def calculate_box(marker_id, rvec, tvec):
    """
    Calculate the box center and smallest enclosing circle.

    Returns:
        {
            "id": marker_id,
            "center": np.array([x, y, z]),
            "radius": float,
        }

    Coordinates are in robot coordinates.
    """

    if marker_id not in BOX_WIDTHS:
        raise ValueError(
            f"No box width defined for marker ID {marker_id}"
        )

    width = BOX_WIDTHS[marker_id]

    # Marker -> camera rotation
    R_camera_marker = rvec_to_matrix(rvec)

    # Marker -> camera transformation
    # The marker origin is at the center of the detected marker.
    T_camera_marker = np.eye(4)
    T_camera_marker[:3, :3] = R_camera_marker
    T_camera_marker[:3,  3] = np.asarray(tvec, dtype=float)

    # Marker -> robot transformation
    T_robot_marker = CAMERA_TO_ROBOT @ T_camera_marker

    R_robot_marker = T_robot_marker[:3, :3]

    # Center of the visible marker/face in robot coordinates.
    face_center_robot = T_robot_marker[:3, 3]

    # Box center:
    # Marker local +Z points OUT of the box.
    # Therefore the box center is:
    #     face_center - normal * depth/2
    # where normal is marker-local +Z expressed in robot coordinates.
    face_normal_robot = R_robot_marker[:, 2]

    box_center_robot = (
        face_center_robot
        - face_normal_robot * (width / 2.0)
    )

    # Construct all 8 corners of the box in marker coordinates.
    # Marker is assumed to be centered on the marked face.
    #   x = left/right across face
    #   y = up/down across face
    #   z = into the box
    # The visible face is z = 0.
    # The back face is z = -depth.

    x_values = [-width / 2.0, width / 2.0]
    y_values = [-width / 2.0, width / 2.0]
    z_values = [0.0, -width]

    corners_marker = np.array([
        [x, y, z]
        for x in x_values
        for y in y_values
        for z in z_values
    ])

    # Transform all corners to robot coordinates
    corners_robot = (
        R_robot_marker @ corners_marker.T
    ).T + face_center_robot

    # Smallest enclosing circle in the robot XY plane
    # The box is centrally symmetric, so its projected center is the
    # center of the minimum enclosing circle.
    # We therefore only need the greatest XY distance from the box center.
    corner_xy = corners_robot[:, :2]
    center_xy = box_center_robot[:2]

    distances = np.linalg.norm(
        corner_xy - center_xy,
        axis=1
    )

    radius = np.max(distances)

    return {
        "id": marker_id,
        "center": box_center_robot,
        "radius": radius,
    }

# ---------------------------------------------------------------------------
# Detect all markers and create the map
# ---------------------------------------------------------------------------

arlo.start_camera()

ids, rvecs, tvecs = arlo.picDetectMarkersPose()

map_array = []

for marker_id, rvec, tvec in zip(ids, rvecs, tvecs):

    marker_id = int(marker_id)

    try:
        box = calculate_box(
            marker_id,
            rvec,
            tvec,
        )

        map_array.append((
            box["id"],
            box["center"],
            box["radius"],
        ))

    except ValueError as e:
        print(e)


# ---------------------------------------------------------------------------
# Result
#
# [
#     (
#         id,
#         np.array([x, y, z]),
#         radius
#     ),
#     ...
# ]
# ---------------------------------------------------------------------------

print("Map:")
for marker_id, center, radius in map_array:
    print(
        f"ID {marker_id}: "
        f"center = {center}, "
        f"radius = {radius:.3f} m"
    )