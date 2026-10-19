import cv2
import picamera2
import numpy as np
from time import sleep

### CONSTANTS
FOCAL_LENGTH = 1288.9 # Camera focal length from Exercise 3.1
CX, CY = 1640/2, 1232/2 # Camera center in pixels
CAMERA_MATRIX = np.array([
    [FOCAL_LENGTH, 0, CX], 
    [0, FOCAL_LENGTH, CY], 
    [0, 0, 1]
], dtype=np.float32) # 3x3 matrix 
DISTORTION_MATRIX = np.zeros((5, 1), dtype=np.float32) # Zero vector ; assumes no camera (lens) distortion

MARKER_SIZE = 0.146 # markørstørrelse i meter på landmarkbox

### Camera Setup
cam = picamera2.Picamera2() #open camera
config = cam.create_video_configuration({ # define camera config suitable for recording video
    "size": (1640, 1232), 
    "format": "RGB888"
}) 

cam.configure(config) #use config
cam.start(show_preview=False) #start camera (turn on)
sleep(1) #wait for camera to start

def take_picture():
    """Takes image in RBG format and return array of shape (hight, width, rbg)"""
    return cam.capture_array("main") #the capture array function captures next image from the stream


### ArUco-setup OpenCV 4.6.0
def picDetectMarkersPose():
    """Takes picture, detects markers in image and estimates poses for detected markers."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

    frame = take_picture()  #hent frame fra PiCamera2, dette er array med shape: (height, width,rbg)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) #laver billed om til gråbilled

    corners, ids, _  = cv2.aruco.detectMarkers(gray, dictionary) #tjekker om vi kan finde nogle Aruco markers fra vores dictionary i billedet corners er hvor markeren er

    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(  #vi beregner translation og rotation
        corners, #vi tager position fra den marker vi ønsker at kører til
        MARKER_SIZE,  #size of marker
        CAMERA_MATRIX, #camera calibration parametre
        DISTORTION_MATRIX
    )
    return ids, rvecs, tvecs


