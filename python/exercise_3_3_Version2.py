import cv2
import cv2.aruco as aruco
import numpy as np
import picamera2

from time import sleep


#kamerakalibrering fra opgave 3.1 ved opløsningen 1640 x 1232.
focal_length = 1288.9       #pixels
cx, cy = 1640 / 2, 1232 / 2 #center of image in pixels
camera_matrix = np.array([[focal_length, 0, cx], 
                          [0, focal_length, cy], 
                          [0, 0, 1]], 
                         dtype=np.float32,)

dist_coeffs = np.zeros((5, 1), dtype=np.float32) 
marker_length = 0.146  #Markeres sorte ramme, har højde 14,6 cm

#ArUco-opsætning til OpenCV 4.6.0 (versionen som ligger på Raspberry Pi'en)
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

#Funktion til at gennem 2D koordinater for landmarkers i et billed
def build_landmark_map(ids, tvecs):
    landmarks = [] #list for landmarks in image
    if ids is None or tvecs is None: #kan der ikke ses nogle markører i billedet returneres en tom liste
        return landmarks

    for marker_id, tvec in zip(ids.flatten(), tvecs): #Hvert markerid og koordinater på markers i billedet
        Xc, Yc, Zc = tvec.reshape(3)
        if np.all(np.isfinite([Xc, Yc, Zc])) and Zc > 0: #Vi tjekker at alle koordinater er tal og at markeren er foran kameraet
            landmarks.append((int(marker_id), (float(Xc), float(Zc))))  #Vi gemmer id og x,z koordinater i listen. 
            
    return landmarks



def main():
    cam = picamera2.Picamera2()
    try:
        config = cam.create_video_configuration(
            {"size": (1640, 1232), "format": "RGB888"},
            queue=False,  #Hent et nyt billede efter brugerens Enter.
        )
        cam.configure(config)
        cam.start(show_preview=False)
        sleep(1)

        while input("Enter: tag billede og udskriv koordinater. q: afslut. ").strip().lower() != "q":
            frame = cam.capture_array("main")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco.detectMarkers(gray, dictionary)

            # Nyt kort for hvert billede. Intet filter på target_id.
            tvecs = None
            if ids is not None and len(ids) > 0:
                _, tvecs, _ = aruco.estimatePoseSingleMarkers(
                    corners, marker_length, camera_matrix, dist_coeffs
                )

            landmarks = build_landmark_map(ids, tvecs)
            print("\nKoordinater i meter fra kameraet: Xc mod højre, Zc fremad")
            for marker_id, (Xc, Zc) in landmarks:
                print(f"ID {marker_id}: Xc = {Xc:.3f} m, Zc = {Zc:.3f} m")
            if not landmarks:
                print("Ingen markører med gyldig position fundet i billedet.")

    except (KeyboardInterrupt, EOFError):
        print("\nKortlægning afsluttet.")
    finally:

        cam.close()


if __name__ == "__main__":
    main()
