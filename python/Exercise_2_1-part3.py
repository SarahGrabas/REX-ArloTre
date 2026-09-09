from time import sleep
import Exercise_1 as E1
import robot

arlo = robot.Robot()

safe_distance = 150
left_speed=60
right_speed=64
turn_speed = 60

print("Obstacle avoidance started")

while True:

    left = arlo.read_left_ping_sensor() > safe_distance
    front = arlo.read_front_ping_sensor() > safe_distance
    right = arlo.read_right_ping_sensor() > safe_distance
    
    vals = (left, front, right)
    
    if vals == (0,0,0):
        arlo.go_diff(left_speed,right_speed,1,1)
        
    elif vals==(0,0,1):
        E1.rotate_inplace(arlo,90,True)
        
    elif vals==(0,1,0):
        E1.rotate_inplace(arlo,90,True)
        
    elif vals==(0,1,1):
        E1.rotate_inplace(arlo,90,True)
        
    elif vals==(1,0,0):
        E1.rotate_inplace(arlo,90,False)
        
    elif vals==(1,0,1):
        E1.rotate_inplace(arlo,180,True)
        
    elif vals==(1,1,0):
        E1.rotate_inplace(arlo,90,False)
        
    elif vals==(1,1,1):
        E1.rotate_inplace(arlo,180,True)
    
    else:
        arlo.stop()