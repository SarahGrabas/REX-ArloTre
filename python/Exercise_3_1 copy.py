import cv2
import time
import picamera2
from PIL import Image
from numpy.typing import NDArray


# Open camera
cam = picamera2.Picamera2()

image_size = (1640, 1232)
config = cam.create_video_configuration(
    {"size": image_size, "format": "RGB888"}
)

cam.configure(config)
cam.start(show_preview=False)

time.sleep(1)


def take_picture(save=False, filename="image.jpg") -> NDArray:
    image = cam.capture_array("main")

    if save: 
        Image.fromarray(image).save(filename)
        # cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    return image


i = 0
while input("Take picture? [y/n]").lower() == "y":
    take_picture(True, f"Pics/{i}.jpg")