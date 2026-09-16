import cv2 # Import the OpenCV library

print("OpenCV version = " + cv2.__version__)

# Open a camera device for capturing
cam = cv2.VideoCapture(0)


if not cam.isOpened(): # Error
    print("Could not open camera")
    exit(-1)
    

while cv2.waitKey(4) == -1: # Wait for a key pressed event
    retval, frameReference = cam.read() # Read frame
    
    if not retval: # Error
        print(" < < <  Game over!  > > > ")
        exit(-1)
    
cv2.imwrite("Images/Image0.png", frameReference)
