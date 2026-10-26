from time import sleep

import cv2
import numpy as np
import serial
from picamera2 import Picamera2


PORT = "/dev/ttyACM0"
MARKER_SIZE = 0.146  
RESOLUTION = (1640, 1232)
# Kalibrering fra tidligere målinger
CAMERA_MATRIX = np.array([
    [1288.9, 0, 820],
    [0, 1288.9, 616],
    [0, 0, 1],
], dtype=np.float64)
DISTORTION = np.zeros(5)


def print_camera_distance(frame):
    """Udskriv afstand til ArUco kodens centrum og vinkel for hver synlig kode"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    if ids is None:
        print("Kamera: ingen ArUco kode fundet")
        return
    
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, MARKER_SIZE, CAMERA_MATRIX, DISTORTION,
    )
    for marker_id, rvec, tvec in zip(ids.flatten(), rvecs, tvecs):
        position = tvec.reshape(3)
        distance = np.linalg.norm(position)
        rotation, _ = cv2.Rodrigues(rvec)
        # Vinkel mellem markørnormalen og synslinjen: 0° er lige forfra.
        cosine = abs(np.dot(rotation[:, 2], position / distance))
        angle = np.degrees(np.arccos(np.clip(cosine, 0, 1)))
        print(f"Kamera ID {marker_id}: {distance * 100:.1f} cm til centrum"
              f" | vinkel: {angle:.1f} grader")


def main():
    camera = Picamera2()
    try:
        with serial.Serial(PORT, 9600, timeout=1, write_timeout=1) as port:
            sleep(2)  # Arduino kan genstarte, når forbindelsen åbnes.
            camera.configure(camera.create_video_configuration(
                main={"size": RESOLUTION, "format": "RGB888"},
            ))
            camera.start(show_preview=False)
            sleep(1)  # Lad kameraets eksponering falde til ro.
            # Picamera2 RGB888 giver BGR-bytes til OpenCV.
            frame = camera.capture_array("main")

            # Sonar aflæses efter billedet
            port.reset_input_buffer()
            port.write(b"0\n")  # Command til frontsonaren
            millimeters = int(port.readline().strip())
            print(f"Sonar (front): {millimeters / 10:.1f} cm")

            print_camera_distance(frame)
    finally:
        camera.close()


if __name__ == "__main__":
    main()
