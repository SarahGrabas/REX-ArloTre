import cv2
import cv2.aruco as aruco
import numpy as np
import picamera2
import Exercise_1 as E1
from robot import arlo
from time import sleep

# Camera setup
cam = picamera2.Picamera2() # Open camera
config = cam.create_video_configuration( # define camera config
    {"size": (1640, 1232), "format": "RGB888"}
)
cam.configure(config) # use config
cam.start(show_preview=False) # start camera (turn on)
sleep(1) # wait for camera to start


def take_picture():
    """Takes image in RBG format and return array of shape (hight, width, rbg)"""
    return cam.capture_array("main")

# 1. Kamerakalibrering (fra opgave 1)
focal_length = 1288.9
cx, cy = 1640/2, 1232/2 # camera center in pixels
camera_matrix = np.array(
    [[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]], dtype=np.float32
)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

marker_length = 0.146  # Markørstørrelse i meter
target_id = 4  # Specifikt ID eller None for enhver markør

# ArUco-opsætning til OpenCV 4.6.0
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)


# Tilstande: "SEARCHING", "ALIGNING", "APPROACHING", "REACHED"
state = "SEARCHING"
lost_frames = 0

while True:
    frame = take_picture()  # Hent frame fra PiCamera2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, _  = aruco.detectMarkers(
        gray, dictionary)

    # Vælg målmarkøren - andre ID'er tæller ikke som et fund
    target_index = None
    if ids is not None and len(ids) > 0:
        if target_id is None:
            target_index = 0
        else:
            matches = np.flatnonzero(ids.flatten() == target_id)
            if matches.size > 0:
                target_index = int(matches[0])

    if target_index is not None:
        lost_frames = 0
        # Vælger første detekterede markør (eller filtrer på target_id)
        # Beregner kun positionen for den valgte markør.
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners[target_index:target_index+1],
            marker_length,
            camera_matrix,
            dist_coeffs
        )


        tvec = tvecs[0][0]  # [Xc, Yc, Zc]
        Xc, Yc, Zc = tvec[0], tvec[1], tvec[2]

        theta = np.arctan2(Xc, Zc)
        distance = Zc

        # Tilstandsovergange & Styring
        if distance <= 0.3:
            state = "REACHED"
            arlo.stop()

        elif abs(theta) > np.radians(5):
            state = "ALIGNING"
            # Omregning: theta er i radianer, rotate_inplace bruger grader.
            # Drejer højst 5 grader, og måler derefter igen
            degrees = min(15.0, float(np.degrees(abs(theta))))
            turn_left = bool(theta < 0)
            E1.rotate_inplace(arlo, degrees, turn_left)
        else:
            state = "APPROACHING"
            # Omregning: distance er i meter, straight_ahead bruger meter.
            # Kører højst 5 cm frem og måler derefter igen.
            step = min(0.20, float(distance - 0.3))
            E1.straight_ahead(arlo, meters=step)
    else:
        arlo.stop()
        lost_frames += 1

        if lost_frames > 5:
            state = "SEARCHING"
            E1.rotate_inplace(arlo, 5, True)
        sleep(0.15)

    if state == "REACHED":
        break
      
