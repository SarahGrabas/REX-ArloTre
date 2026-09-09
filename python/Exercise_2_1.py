from time import sleep
import robot

arlo = robot.Robot()

safe_distance = 150
forward_speed = 64
turn_speed = 60

print("Obstacle avoidance started")

while True:

    left = arlo.read_left_ping_sensor()
    sleep(0.05)

    front = arlo.read_front_ping_sensor()
    sleep(0.05)

    right = arlo.read_right_ping_sensor()
    sleep(0.05)

    print("Left:", left, "Front:", front, "Right:", right)

    if front < safe_distance:

        arlo.stop()
        sleep(0.1)

        if left > right:
            print("Turning left")
            arlo.go_diff(turn_speed, turn_speed, 0, 1)

        else:
            print("Turning right")
            arlo.go_diff(turn_speed, turn_speed, 1, 0)

        sleep(0.5)

    elif left < safe_distance:
        print("Obstacle on left - steering right")
        arlo.go_diff(70, 50, 1, 1)

    elif right < safe_distance:
        print("Obstacle on right - steering left")
        arlo.go_diff(50, 70, 1, 1)

    else:
        print("Driving forward")
        arlo.go_diff(forward_speed-3, forward_speed, 1, 1)

    sleep(0.1)