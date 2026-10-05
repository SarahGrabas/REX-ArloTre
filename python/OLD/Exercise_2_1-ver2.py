import robot
from time import sleep
import random
    
def random_direction(arlo, leftSpeed, rightSpeed):
    Left_or_Right = print(random.randrange(0, 2))
    Angle_rand = print(random.randrange(81, 243)) 
    #81 svarer til en kvart omgang, altså 9.8 s/3/4
    #326 svarer til en 3/4 omgang, altså 81*(3/4)
    
    if Left_or_Right==0: #turn right
        arlo.go_diff(leftSpeed, rightSpeed, 1, 0)
        sleep(Angle_rand*0.01) 

    if Left_or_Right==1: #turn right
        arlo.go_diff(leftSpeed, rightSpeed, 0, 1)
        sleep(Angle_rand*0.01) 


arlo = robot.Robot()

safe_distance = 300
leftSpeed = 64
rightSpeed = 64



arlo_run=True    
while arlo_run:

        left = arlo.read_left_ping_sensor()
        sleep(0.05)

        front = arlo.read_front_ping_sensor()
        sleep(0.05)

        right = arlo.read_right_ping_sensor()
        sleep(0.05)
        
        if left==-1 or right==-1 or front==-1:
                arlo_run=False
                print("fejl ved sensor")
                

        if front < safe_distance:
                sleep(0.1)
                random_direction(arlo, leftSpeed, rightSpeed)
                
        if left < safe_distance:
            arlo.go_diff(leftSpeed, rightSpeed, 0, 1) #turn left
            sleep(81*0.01) #81*0.01 er en kvart omgang
            
        if right< safe_distance:
            arlo.go_diff(leftSpeed, rightSpeed, 1, 0) #turn right
            sleep(81*0.01) #81*0.01 er en kvart omgang

        # elif left < safe_distance and right < safe_distance and front < safe_distance:
        #         back = arlo.read_back_ping_sensor()
        #         sleep(0.05)
                
        #         if back <safe_distance:
        #                 arlo.go_diff(leftSpeed, rightSpeed, 0, 0) #bak
        #                 sleep(3) #lidt over en meter
        #                 random_direction(arlo, leftSpeed, rightSpeed) #Og vælg ny random retning

        #         else:
        #                 random_direction(arlo, leftSpeed, rightSpeed) #Vælg random direction indtil den kan
        else: 
                arlo.go_diff(leftSpeed, rightSpeed, 1, 1) 
                        
    
#arlo.go_diff(leftSpeed=61, rightSpeed=64, 1, 1), 5 sekunder = 2 meter
#arlo.go_diff(leftSpeed=61, rightSpeed=64, 0, 1), 9.8 sekunder = 3 omgange (venstre rotation)
    
            
            
        
        
    
