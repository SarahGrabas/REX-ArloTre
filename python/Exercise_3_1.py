from picamera2 import Picamera2
import time

picam2 = Picamera2()

# Konfigurer kameraet til stillbilleder
config = picam2.create_still_configuration(
    main={"size": (1640, 1232)}
)
picam2.configure(config)

# Start kameraet
picam2.start()

# Giv kameraet lidt tid til at stabilisere eksponering/farver
time.sleep(2)

# Tag billedet
i = 0
while input("Take picture? [y/n]").lower() == "y":
    picam2.capture_file(f"{i}.jpg")
    i += 1

# Luk kameraet
picam2.stop()
picam2.close()

# Use the following bash command to transfer image via SSH:
# scp pi@192.168.98.102:~/REX-ArloTre/python/billede.jpg .