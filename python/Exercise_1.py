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

def continuous(arlo:Robot, n=1, swap=False, sleeps=(9,8)):
    l,r = sleeps
    for _ in range(n):
        if swap:
            print(arlo.go_diff(100, 55, 1, 1))
            sleep(r)
            print(arlo.go_diff(55, 100, 1, 1))
            sleep(l)
        else:
            print(arlo.go_diff(55, 100, 1, 1))
            sleep(l)
            print(arlo.go_diff(100, 55, 1, 1))
            sleep(r)
    print(arlo.stop())

### Main ###

if __name__ == "__main__":
    from robot import arlo

    # moter warmup
    arlo.go_diff(50, 50, 1, 0)
    sleep(1)
    arlo.go_diff(50, 50, 0, 1)
    sleep(1)
    arlo.stop()

    while True:
        userinput = input("""
Enter number for desired action [1/2/3/4/<Enter>] :
[0] Calibration
[1] Straigt ahead
[2] Rotate inplace
[3] Square shape
[4] Eight shape (continuous)
<Enter> Terminate
> """)
        match userinput:
            case '0':
                print("Calibration")

                c0_left_wheel = True
                c0_forward_drive = True
                c0_speed = 40
                c0_divisor = 4
                c0_wait = 0.5
                c0_n = 1

                print("Configure parameters (press <Enter> for default):")
                while True:
                    userinput = input(f"> left_wheel (bool | default {c0_left_wheel}) = ")
                    try:
                        c0_left_wheel = bool(userinput)
                        break
                    except:
                        print("Invalid input!")
                        continue
                while True:
                    userinput = input(f"> forward_drive (bool | default {c0_forward_drive}) = ")
                    try:
                        c0_forward_drive = bool(userinput)
                        break
                    except:
                        print("Invalid input!")
                        continue
                while True:
                    userinput = input("> speed (int | range {0, [40;127]} "+f"| default {c0_speed}) = ")
                    try:
                        c0_speed = int(userinput)
                        if Robot._power_checker(c0_speed): 
                            break
                        else:
                            print("Invalid input!")
                            continue
                    except:
                        print("Invalid input!")
                        continue
                while True:
                    userinput = input(f"> divisor (int | degrees = 360/divisor | default {c0_divisor}) = ")
                    try:
                        c0_divisor = int(userinput)
                        if c0_divisor > 0: 
                            break
                        else:
                            print("Invalid input!")
                            continue
                    except:
                        print("Invalid input!")
                        continue
                while True:
                    userinput = input(f"> wait (float | default {c0_wait}) = ")
                    try:
                        c0_wait = float(userinput)
                        if c0_wait > 0: 
                            break
                        else:
                            print("Invalid input!")
                            continue
                    except:
                        print("Invalid input!")
                        continue
                while True:
                    userinput = input(f"> n (int | default {c0_n}) = ")
                    try:
                        c0_n = int(userinput)
                        if c0_n > 0: 
                            break
                        else:
                            print("Invalid input!")
                            continue
                    except:
                        print("Invalid input!")
                        continue
                
                c0_results = []
                
                c0_lower = 0.0
                c0_middle = 1.0
                c0_upper = None
                
                for _ in range(c0_divisor*c0_n):
                    sleep(c0_wait)
                    arlo.go_diff(c0_speed*c0_left_wheel, c0_speed*(not c0_left_wheel), c0_forward_drive, c0_forward_drive)
                    sleep(c0_middle)
                    arlo.stop()
                
                while True:
                    userinput = input("""
                Enter adjustment action [-/0/+/<Enter>]:
                [?] Again
                [-] Decrease
                [0] Spot on
                [+] Increase
                <Enter> Abort
                > """)
                    match userinput:
                        case '?':
                            pass
                        case '-':
                            c0_upper = c0_middle
                            c0_middle = (c0_lower+c0_upper)/2
                        case '0':
                            c0_results.insert(0, {
                                "left_wheel": c0_left_wheel,
                                "forward_drive": c0_forward_drive,
                                "speed": c0_speed,
                                "divisor": c0_divisor,
                                "degrees": 360/c0_divisor,
                                "wait": c0_wait,
                                "n": c0_n,
                                "result_time": c0_middle
                            })
                            print(c0_results)
                            break
                        case '+':
                            c0_lower = c0_middle
                            c0_middle = c0_middle*2 if c0_upper is None else (c0_lower+c0_upper)/2
                        case '':
                            break
                        case _:
                            print("Invalid input!")
                            continue
                        
                    for _ in range(c0_divisor*c0_n):
                        sleep(c0_wait)
                        arlo.go_diff(c0_speed*c0_left_wheel, c0_speed*(not c0_left_wheel), c0_forward_drive, c0_forward_drive)
                        sleep(c0_middle)
                        arlo.stop()

            case '1':
                print("Straigt ahead:")

                c1_meters = 1.0
                print("Configure parameters (press <Enter> for default):")
                while True:
                    userinput = input(f"> meters (float | default {c1_meters}) = ")
                    try:
                        c1_meters = float(userinput)
                        if c1_meters >= 0: 
                            break
                        else:
                            print("Invalid input!")
                            continue
                    except:
                        print("Invalid input!")
                        continue
                straight_ahead(arlo, c1_meters)

            case '2':
                print("Rotate inplace:")

                c2_degrees = 90.0
                c2_turn_left = True
                print("Configure parameters (press <Enter> for default):")
                while True:
                    userinput = input(f"> degrees (float | default {c2_degrees}) = ")
                    try:
                        c2_degrees = float(userinput)
                        if c2_degrees >= 0: 
                            break
                        else:
                            print("Invalid input!")
                            continue
                    except:
                        print("Invalid input!")
                        continue
                while True:
                    userinput = input(f"> turn_left (bool | default {c2_turn_left}) = ")
                    try:
                        c2_turn_left = bool(userinput)
                        break
                    except:
                        print("Invalid input!")
                        continue
                rotate_inplace(arlo, c2_degrees, c2_turn_left)
                
            case '3':
                print("Square shape:")

                c3_side_length = 1.0,
                c3_turn_left = True,
                c3_angle = 90.0
                print("Configure parameters (press <Enter> for default):")
                while True:
                    userinput = input(f"> side_length (float | default {c3_side_length}) = ")
                    try:
                        c3_side_length = float(userinput)
                        if c3_side_length >= 0: 
                            break
                        else:
                            print("Invalid input!")
                            continue
                    except:
                        print("Invalid input!")
                        continue
                while True:
                    userinput = input(f"> turn_left (bool | default {c3_turn_left}) = ")
                    try:
                        c3_turn_left = bool(userinput)
                        break
                    except:
                        print("Invalid input!")
                        continue
                while True:
                    userinput = input(f"> angle (float | default {c3_angle}) = ")
                    try:
                        c3_angle = float(userinput)
                        if c3_angle >= 0: 
                            break
                        else:
                            print("Invalid input!")
                            continue
                    except:
                        print("Invalid input!")
                        continue
                square(arlo, c3_side_length, c3_turn_left, c3_angle)

            case '4':
                print("Eight shape (continuous):")

                c4_n = 1,
                c4_swap = False,
                c4_left_sleep = 9.0
                c4_right_sleep = 8.0
                print("Configure parameters (press <Enter> for default):")
                while True:
                    userinput = input(f"> n (int | default {c4_n}) = ")
                    try:
                        c4_n = int(userinput)
                        if c4_n > 0: 
                            break
                        else:
                            print("Invalid input!")
                            continue
                    except:
                        print("Invalid input!")
                        continue
                while True:
                    userinput = input(f"> swap (bool | default {c4_swap}) = ")
                    try:
                        c4_swap = bool(userinput)
                        break
                    except:
                        print("Invalid input!")
                        continue
                while True:
                    userinput = input(f"> left_sleep (float | default {c4_left_sleep}) = ")
                    try:
                        c4_left_sleep = float(userinput)
                        if c4_left_sleep >= 0: 
                            break
                        else:
                            print("Invalid input!")
                            continue
                    except:
                        print("Invalid input!")
                        continue
                while True:
                    userinput = input(f"> right_sleep (float | default {c4_right_sleep}) = ")
                    try:
                        c4_right_sleep = float(userinput)
                        if c4_right_sleep >= 0: 
                            break
                        else:
                            print("Invalid input!")
                            continue
                    except:
                        print("Invalid input!")
                        continue
                continuous(arlo, c4_n, c4_swap, (c4_left_sleep, c4_right_sleep))

            case _:
                break
