# A file to be changed if needed for called functions from
# non-selfcontained Exercise solution files, or for other
# temporary ArloTre relevant code. 

import Exercise_1 as E1
from time import sleep
from robot import arlo

# arlo._calibrate_speed(1, 1, 3, 360, 8)

# left, forward, deg 360, 8 sec, 51
# left, backward, deg 360, 8 sec, 51
# right, forward, deg 360, 8 sec, 57
# right, backward, deg 360, 8 sec, 57

while True:
    usr = input("speed = ")
    for _ in range(3):
        sleep(0.5)
        arlo.go_diff(0, int(usr), 1, 1)
        sleep(8)
        arlo.stop()