import cv2
import numpy as np
import time
from picamera2 import Picamera2

# Camera
picam2 = Picamera2()

config = picam2.create_still_configuration(
    main={"size": (1640, 1232)}
)

picam2.configure(config)
picam2.start()

time.sleep(2)

# ArUco
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Camera calibration
focal_length = 1288.9
marker_size = 0.146

camera_matrix = np.array([
    [focal_length, 0, 820],
    [0, focal_length, 616],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.zeros((5, 1))


while input("Take measurement? [y/n] ").lower() == "y":
    frame = picam2.capture_array()

    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict)

    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            marker_size,
            camera_matrix,
            dist_coeffs
        )

        for tvec in tvecs:
            distance = np.linalg.norm(tvec[0])
            print(f"Distance: {distance:.3f} m")

picam2.stop()
picam2.close()