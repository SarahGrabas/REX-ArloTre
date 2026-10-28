import rally
import RRT
import selflocalize
import numpy as np
import time

arlo = rally.Robot()
arlo_radius =rally.ARLO_RADIUS *100 #fra meter til cm

safe_distance_to_lm = 0.3 #m
safe_distance_to_obstacles = 0.3 #m

valid_objects=[8,9,10,11]
valid_coords={
    8: (131, 131),
    9: (131,431 ), #466
    10: (531 , 131), #566
    11: (531 , 431)
}

#MAP
# X_MIN= 0.0
# X_MAX =  690 
# Y_MIN= 0.0
# Y_MAX = 590

def sonars_choose_direction(current_state,arlo):
    left = arlo.sonars.left()
    front = arlo.sonars.front()
    right = arlo.sonars.right()
    back = arlo.sonars.back()
    
    

    distances = {
        "left": left,
        "front": front,
        "right": right,
        "back": back
    }

    # Find retningen med størst fri afstand
    direction = max(distances, key=distances.get)

    print("Sonars:")
    print("  Left: ", left)
    print("  Front:", front)
    print("  Right:", right)
    print("  Back: ", back)
    print("Choosing:", direction)

    # Drej mod den valgte retning
    if direction == "left":
        arlo.motors.rotate(90)
        current_state["movement"][1]=90

    elif direction == "right":
        arlo.motors.rotate(-90)
        current_state["movement"][1]=-90

    elif direction == "back":
        arlo.motors.rotate(180)
        current_state["movement"]=180
        
    

    # Hvis front er bedst, skal vi ikke dreje

    # Kør 30 cm
    drive_with_sonars(current_state,arlo,50, True)

    return direction

def drive_with_sonars(current_state,arlo,cm, forward=True):

    meters = cm / 100

    start = time.time()

    # Beregn hvor lang tid det burde tage
    drive_time = meters / arlo.motors.meters_pr_sec

    # Start motorerne uden at vente
    arlo.motors.drive(
        meters,
        forward,
        _sleep=False,
        stop_when_done=False
    )

    while True:
        current_time = time.time()

        left = arlo.sonars.left() < safe_distance_to_obstacles
        front = arlo.sonars.front() < safe_distance_to_obstacles
        right = arlo.sonars.right() < safe_distance_to_obstacles

        # Stop hvis der er en forhindring
        if left or front or right:
            arlo.motors.stop()
            break

        # Stop når den ønskede tid er gået
        if current_time - start >= drive_time:
            arlo.motors.stop()
            break

        time.sleep(0.01)

    total_driven_time = time.time() - start
    meter_driven = arlo.motors.meters_pr_sec * total_driven_time

    print("Centimeters driven:", meter_driven * 100)
    current_state["movement"][0]+=meter_driven * 100

    return meter_driven * 100

def execute_robot_path(current_state,arlo,path,number_nodes:int):
    """Funktion til at køre den simpler path
    Gridsize er længde på x og y akse i cm"""

    executed_path = [path[0]]
    
    path_length=len(path)
    
    if number_nodes > path_length: 
        number_nodes=path_length

    current_angle = 0  # Robotten starter med at pege ligeud
    
    for i in range(number_nodes - 1): #Vi kører kun til antal noder angivet

        x1, y1 = path[i] #from node
        x2, y2 = path[i + 1]# til node

        dx = x2 - x1
        dy = y2 - y1

        distance = np.sqrt(dx**2 + dy**2)

        desired_angle = -np.degrees(np.arctan2(dx, dy))

        relative_angle = desired_angle - current_angle
        
        print("         desired_angle, relative_angle, distance",
              desired_angle, relative_angle, distance)

        arlo.motors.rotate(relative_angle)
        current_state["movement"][1]+=relative_angle
        drive_with_sonars(current_state,arlo,distance) #input i cm

        current_angle = desired_angle
        
        executed_path.append([x2, y2])

    arlo.motors.stop()
    return executed_path


phases=[8,9,10,11,8]
current_goal_index=0

current_state=selflocalize.initialize_mcl(num_particles=1500)

while True:  
    
    valid_lm_seen =[] #Vi nulstiller efter selflocalization loop
    scan_counter=1
    while len(valid_lm_seen)<2: #Vi skal have set 2 valid landmark for selflocalization, før vi stoler på estimated robot pose
        if scan_counter%5==0: #Ved hver 5 scan bevæger vi os lidt, hvis vi ikke kan se 2 lms
            sonars_choose_direction(current_state,arlo) #når vi bevæger os, opdateres current state inde i funktionen

        print("Selflocalizing")
        landmarks_seen = arlo.camera.detect_landmarks()
        
        all_obj_in_frame, valid_lm_seen=selflocalize.data_objects(landmarks_seen, valid_lm_seen, valid_objects)  
        selflocalize.mcl_update(current_state,all_obj_in_frame,valid_objects,valid_coords)

        selflocalize.scan_function(current_state,arlo)
        
        scan_counter+=1

     
    GOAL_ID=phases[current_goal_index]
    GOAL_COORDS=valid_coords[phases[current_goal_index]] 
    
    
    lm_radius = np.sqrt(rally.Landmark.MAX_BOX_SIZES[GOAL_ID]**2 / 2) #m
    reach_dist = (lm_radius + rally.ARLO_RADIUS + safe_distance_to_lm ) * 100 #fra meter til cm

    dist_to_goal = np.hypot(current_state["est_pose"].getX() - GOAL_COORDS[0], current_state["est_pose"].getY() - GOAL_COORDS[1])

    if dist_to_goal <= reach_dist:
        print("Goal reached")
        current_goal_index += 1
        GOAL_ID=phases[current_goal_index]
        GOAL_COORDS=valid_coords[phases[current_goal_index]] 
        
        if current_goal_index == len(phases):
            break  # rally færdigt
        
    (x,y,theta) = (current_state["est_pose"].getX(),current_state["est_pose"].getY(),current_state["est_pose"].getTheta())
        
    obstacles=RRT.convert_lm_center_to_global((x,y,theta),landmarks_seen, GOAL_ID) #denne funktion håndterer landmark koordinater i robot koords, og assumer at dubletter er fjernet 
        
    print(f"RRT TO {GOAL_ID}")
    
    path=RRT.RRT_call(GOAL_COORDS, obstacles , current_state["est_pose"], arlo_radius)
        
    if path: #ikke er none
        print("Path found: ", path)
        number_nodes=len(path) #antal noder fra path vi ønsker at kører til
        execute_robot_path(current_state,arlo,path,number_nodes)
            #Vi mangler at putte sensors på dette
            #Her kaldes en funktion der kører til første punkt i path med sensorer tændt mens den køre
        
    elif path is None:
        print(f"Could not find path to {GOAL_ID}")
            #Vi skal lave en funktion, der bruger sensorerne til at køre lidt væk fra obstacles, så vi kan scanne igen.
            
        #skal vi køre lidt i en retning væk fra obstacles og scanner igen
        sonars_choose_direction(current_state,arlo)
        


#Vi skal have en funktion, der beslutter om vi er ved goal ved at bruge sensor og kamera

# def sensor_choose_way():
#     left = arlo.read_left_ping_sensor() < safe_distance_to_obstacles
#     front = arlo.read_front_ping_sensor() < safe_distance_to_obstacles
#     right = arlo.read_right_ping_sensor() < safe_distance_to_obstacles
#     back = arlo.read_back_ping_sensor() < safe_distance_to_obstacles
    
#     vals = (left, front, right)
#     print(vals)
    
#     if vals == (0,0,0):
#         arlo.move() #20 cm maybe, hvor der stadig intet er
        
#     elif vals==(0,0,1):
#         #E1.rotate_inplace(arlo,90,True)
        
#     elif vals==(0,1,0):
#         #E1.rotate_inplace(arlo,90,True)
        
#     elif vals==(0,1,1):
#         #E1.rotate_inplace(arlo,90,True)
        
#     elif vals==(1,0,0):
#         #E1.rotate_inplace(arlo,90,False)
        
#     elif vals==(1,0,1):
#         #E1.rotate_inplace(arlo,180,True)
        
#     elif vals==(1,1,0):
#         #E1.rotate_inplace(arlo,90,False)
        
#     elif vals==(1,1,1):
#         #E1.rotate_inplace(arlo,180,True)
    
#     else:
#         #arlo.stop()