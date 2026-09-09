from robot import Robot
from time import sleep

### Part 1 ###

def straight_ahead(arlo:Robot, meters=1.0):
    sec_pr_meter = 5 / 2
    print(arlo.go_diff(61, 64, 1,1))
    sleep(meters * sec_pr_meter)
    print(arlo.stop())

def rotate_inplace(arlo:Robot, degrees=90.0, turn_left=True):
    if turn_left:
        sec_pr_degree = 9.8 / (3*360)
        print(arlo.go_diff(61, 64, 0, 1))
        sleep(degrees * sec_pr_degree)
    else:
        sec_pr_degree = 9.8 / (3*360)
        print(arlo.go_diff(61, 64, 1, 0))
        sleep(degrees * sec_pr_degree)
    print(arlo.stop())

def square(arlo:Robot, side_length=1.0, turn_left=True, angle=90.0, old=False):
    if old:
    	for _ in range(4):
            print(arlo.go_diff(60, 64, 1,1))
            sleep(2.5)
            print(arlo.go_diff(61, 64, 0,1))
            sleep(0.7975)
            sleep(0.1)
    else:
        for _ in range(4):
            straight_ahead(arlo, side_length)
            rotate_inplace(arlo, degrees=angle, turn_left=turn_left)



### Part 2 ###

def continuous(arlo:Robot, n=1):
    for _ in range(n):
        print(arlo.go_diff(55, 100, 1, 1))
        sleep(7)
        print(arlo.go_diff(100,55, 1, 1))
        sleep(7)
    print(arlo.stop())