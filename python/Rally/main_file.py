import rally
import RRT
import selflocalize
import numpy as np

arlo = rally.Robot()
arlo_radius =rally.ARLO_RADIUS *100 #fra meter til cm

safe_distance_to_lm = 350 #mm
safe_distance_to_obstacles = 300 #mm

valid_objects=[1,2,3,6]
valid_coords={
    1: (131, 131),
    2: (131,431 ), #466
    3: (531 , 131), #566
    6: (531 , 431)
}

#MAP
# X_MIN= 0.0
# X_MAX =  690 
# Y_MIN= 0.0
# Y_MAX = 590

phases=[1,2,3,4,1]
current_goal_index=0

current_state=selflocalize.initialize_mcl(num_particles=1500)

while True:  
    
    valid_lm_seen =[] #Vi nulstiller efter selflocalization loop
    while len(valid_lm_seen)<2: #Vi skal have set 2 valid landmark for selflocalization, før vi stoler på estimated robot pose
        print("Selflocalizing")
        landmarks_seen = arlo.camera.detect_landmarks()
        
        all_obj_in_frame, valid_lm_seen=selflocalize.data_objects(landmarks_seen, valid_lm_seen, valid_objects)  
        selflocalize.mcl_update(current_state,all_obj_in_frame,valid_objects,valid_coords)

        selflocalize.scan_function(current_state,arlo)
    
    
    
#     if phases[current_goal] in all_obj_in_frame: #hvis vi kan se current goal, så laver vi RRT til den
#         GOAL_ID=phases[current_goal]
#         GOAL_COORDS=valid_coords[phases[current_goal]]
        
#         obstacles=RRT.convert_lm_center_to_global(landmarks_seen, GOAL_ID) #denne funktion håndterer landmark koordinater i robot koords, og assumer at dubletter er fjernet 
        
#         print(f"RRT TO {GOAL_ID}")
#         path=RRT.RRT_call(GOAL_COORDS, obstacles , current_state["est_pose"], arlo_radius)
        
        
#         if path: #ikke er none
        
#             number_nodes=len(path) #antal noder fra path vi ønsker at kører til
#             RRT.execute_robot_path(arlo,path,number_nodes)
#             #Vi mangler at putte sensors på dette
#             #Her kaldes en funktion der kører til første punkt i path med sensorer tændt mens den køre
        
#         elif path is None:
#             print(f"Could not find path to {GOAL_ID}")
#             #Vi skal lave en funktion, der bruger sensorerne til at køre lidt væk fra obstacles, så vi kan scanne igen.
            
#             #skal vi køre lidt i en retning væk fra obstacles og scanner igen
        
#     else: #roter så vi kan se current goal 
        
    
#     if #sensorer registrere noget inden for 40 cm, så stop kørsel, selflocalize og forfra
        
    
#     if :#betingelse for at robot har været ved current goal
#         current_goal+=1

# #Vi skal have en funktion, der beslutter om vi er ved goal ved at bruge sensor og kamera

# def sensor_choose_way():
#     left = arlo.read_left_ping_sensor() < safe_distance_to_obstacles
#     front = arlo.read_front_ping_sensor() < safe_distance_to_obstacles
#     right = arlo.read_right_ping_sensor() < safe_distance_to_obstacles
    
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