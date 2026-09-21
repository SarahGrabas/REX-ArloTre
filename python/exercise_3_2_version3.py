import cv2
import cv2.aruco as aruco
import numpy as np
import picamera2
import Exercise_1 as E1
from robot import arlo
from time import sleep

#camera setup
cam = picamera2.Picamera2() #open camera
config = cam.create_video_configuration( {"size": (1640, 1232), "format": "RGB888"}) #define camera config suitable for recording video

cam.configure(config) #use config
cam.start(show_preview=False) #start camera (turn on)
sleep(1) #wait for camera to start


def take_picture():
    """Takes image in RBG format and return array of shape (hight, width, rbg)"""
    return cam.capture_array("main") #the capture array function captures next image from the stream

#kamerakalibrering (fra opgave 1)
focal_length = 1288.9
cx, cy = 1640/2, 1232/2 #camera center in pixels
camera_matrix = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]], dtype=np.float32) #3x3 matrix 

dist_coeffs = np.zeros((5, 1), dtype=np.float32) #zero-vector with 5 rows 

marker_length = 0.146  #markørstørrelse i meter på landmarkbox
target_id = 4  #specifikt ID eller None for enhver markør

#ArUco-opsætning til OpenCV 4.6.0
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

#states: "SEARCHING", "ALIGNING", "APPROACHING", "REACHED"
state = "SEARCHING"
lost_frames = 0

while True:
    frame = take_picture()  #hent frame fra PiCamera2, dette er array med shape: (height, width,rbg)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #laver billed om til gråbilled

    corners, ids, _  = aruco.detectMarkers(gray, dictionary) #tjekker om vi kan finde nogle Aruco markers fra vores dictionary i billedet
                                                            #corners er hvor markeren er

    #vælg målmarkøren når vi kan se flere markers
    target_index = None #None når der ikke er markers
    if ids is not None and len(ids) > 0: #vi ser mindst 1 marker
        if target_id is None: #vi har ikke én bestemt marker vi leder efter, så vi bruger den første vi ser
            target_index = 0 
        else:                   #hvis vi har en target marker
            matches = np.flatnonzero(ids.flatten() == target_id) #vi tjekker om nogle af de markers vi ser i billedet matcher med vores target marker
            if matches.size > 0: #Hvis vi har mindst ét match, så tager vi første match og gemmer
                target_index = int(matches[0])

    if target_index is not None: #vi har en marker i billedet som vi gerne vil hen til.
        lost_frames = 0
    
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(  #vi beregner translation og rotation
            corners[target_index:target_index+1], #vi tager position fra den marker vi ønsker at kører til
            marker_length,  #size of marker
            camera_matrix, #camera calibration parametre
            dist_coeffs
        )


        tvec = tvecs[0][0]  #kameraets koordinatsystem
        Xc, Yc, Zc = tvec[0], tvec[1], tvec[2]

        theta = np.arctan2(Xc, Zc) #vi beregner vinklen mellem robottens fremadgående retning og markeren
        distance = np.linalg.norm([Xc, Zc]) # Ændret fra Zc til nu at beregne afstanden. sqrt(Xc^^2+Zc^^2)

        if distance <= 0.3: #Hvis robotten er 30 cm tæt på markeren behøver vi ikke kører mod den, vi har REACHED den
            state = "REACHED"
            arlo.stop()

        elif abs(theta) > np.radians(5):    #hvis markeren er mere end 5 grader væk, drejer vi højst 5 grader og måler igen
            state = "ALIGNING"
            degrees = min(5.0, float(np.degrees(abs(theta)))) #Hvis vi er mindre end 5 grader fra, så drejere vi det antal grader vi mangler
            turn_left = bool(theta < 0) #hvis theta er negativ så drejer vi til højre, fordi markøren så er til højre for robottens x-aksen
                                        #hvis theta er positiv så drejer vi til venstre.
            E1.rotate_inplace(arlo, degrees, turn_left)
            
        else:                               #Hvis robottens retning og markøren er mindre eller lig 5 grader, begynder vi at approache den
            state = "APPROACHING"
            step = min(0.20, float(distance - 0.3)) #distance er i meter, straight_ahead bruger meter.
            E1.straight_ahead(arlo, meters=step) #Vi kører højst 20 cm frem, indtil vi er 30 cm fra markeren.
            
    else:       #Hvis markeren ikke kan ses, lægger vi én til lost frames
        arlo.stop()
        lost_frames += 1

        if lost_frames > 5:     #Hvis vi har haft mere end 5 frames uden en marker i sigte, så roterer vi.
            state = "SEARCHING"
            E1.rotate_inplace(arlo, 15, True)
        sleep(0.15)

    if state == "REACHED":
        break
      
