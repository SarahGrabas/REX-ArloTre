# A file to be changed if needed for called functions from
# non-selfcontained Exercise solution files, or for other
# temporary ArloTre relevant code. 

import Exercise_1 as E1
from robot import arlo

while input("\nDo action? [y/n] ").lower() == "y":

    E1.square(arlo, turn_left=True, angle=float(input("angle=")))

    E1.continuous(arlo, n=int(input("n=")), swap=bool(int(input("swap="))), times=(float(input("l=")),float(input("r="))))