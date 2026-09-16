import cv2
import cv2.aruco as aruco
import numpy as np
import picamera2
from robot import arlo

# Camera setup
cam = picamera2.Picamera2() # Open camera
config = cam.create_video_configuration( # define camera config
    {"size": (1640, 1232), "format": "RGB888"}
)
cam.configure(config) # use config
cam.start(show_preview=False) # start camera (turn on)
time.sleep(1) # wait for camera to start


def take_picture():
    """Takes image in RBG format and return array of shape (hight, width, rbg)"""
    return cam.capture_array("main")

# 1. Kamerakalibrering (fra opgave 1)
focal_length = 12.889
cx, cy = 1640/2, 1232/2 # camera center in pixels
camera_matrix = np.array(
    [[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]], dtype=np.float32
)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

marker_length = 0.146  # Markørstørrelse i meter
target_id = 4  # Specifikt ID eller None for enhver markør

# ArUco Opsætning
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

# Tilstande: "SEARCHING", "ALIGNING", "APPROACHING", "REACHED"
state = "SEARCHING"
lost_frames = 0

while True:
    frame = take_picture()  # Hent frame fra PiCamera2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = aruco.detectMarkers(
        gray, dictionary, parameters=parameters
    )

    if ids is not None and len(ids) > 0:
        lost_frames = 0
        # Vælg første detekterede markør (eller filtrer på target_id)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, marker_length, camera_matrix, dist_coeffs
        )

        tvec = tvecs[0][0]  # [Xc, Yc, Zc]
        Xc, Yc, Zc = tvec[0], tvec[1], tvec[2]

        theta = np.arctan2(Xc, Zc)
        distance = Zc

        # Tilstandsovergange & Styring
        if distance <= 0.3:  # Stopafstand 30 cm
            state = "REACHED"
            set_robot_velocities(v=0, omega=0)
        elif abs(theta) > np.radians(5):
            state = "ALIGNING"
            omega = -1.5 * theta  # P-regulator vinkel
            set_robot_velocities(v=0, omega=omega)
        else:
            state = "APPROACHING"
            v = 0.2 * (distance - 0.3)  # P-regulator fremdrift
            omega = -1.0 * theta
            set_robot_velocities(v=v, omega=omega)
    else:
        lost_frames += 1
        if lost_frames > 5:
            state = "SEARCHING"
            set_robot_velocities(v=0, omega=0.4)  # Roter langsomt om egen akse

    if state == "REACHED":
        break
