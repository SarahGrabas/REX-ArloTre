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

def square(arlo:Robot, side_length:1.0, turn_left=True):
    # 1
    straight_ahead(arlo, side_length)
    rotate_inplace(arlo, turn_left=turn_left)
    # 2
    straight_ahead(arlo, side_length)
    rotate_inplace(arlo, turn_left=turn_left)
    # 3
    straight_ahead(arlo, side_length)
    rotate_inplace(arlo, turn_left=turn_left)
    # 4
    straight_ahead(arlo, side_length)
    rotate_inplace(arlo, turn_left=turn_left)



### Part 2 ###

def continuous(arlo:Robot, n=1):
    for _ in range(n):
        print(arlo.go_diff(45, 90, 1, 1))
        sleep(7)
        print(arlo.go_diff(90,45, 1, 1))
        sleep(7)
    print(arlo.stop())
