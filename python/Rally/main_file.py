import rally
import RRT
import RRT_grid
import selflocalize
import numpy as np

arlo = rally.Robot()
arlo_radius =rally.ARLO_RADIUS

phases=[1,2,3,4,1]
current_goal_index=0

current_state=selflocalize.initialize_mcl(num_particles=1500)

while True:   
    valid_lm_seen =[] #Vi nulstiller efter selflocalization loop
    while len(valid_lm_seen)<2: #Vi skal have set 2 valid landmark for selflocalization, før vi stoler på estimated robot pose
        landmarks_seen = arlo.camera.detect_landmarks()

        valid_objects =[] #dem vi vil selflocalize ud fra
        
        all_obj_in_frame, valid_coords, valid_lm_seen=selflocalize.sorting_objects(valid_objects,landmarks_seen, valid_lm_seen) #denne funktion assumer at dubletter er fjernet 
        selflocalize.mcl_update(current_state,all_obj_in_frame,valid_objects,valid_coords)

        selflocalize.scan_function(current_state,arlo)
    
    
    
    if phases[current_goal] in all_obj_in_frame: #hvis vi kan se current goal, så laver vi RRT til den
        GOAL_ID=phases[current_goal]
        
        GOAL,obstacles=RRT.convert_lm_center_to_global(landmarks_seen, GOAL_ID) #denne funktion håndterer landmark koordinater i robot koords, og assumer at dubletter er fjernet 
        path=RRT.RRT_call(GOAL, obstacles , current_state["est_pose"], arlo_radius)
        
        number_nodes=len(path) #antal noder fra path vi ønsker at kører til
        RRT.execute_robot_path(arlo,path,number_nodes)
        #Vi mangler at putte sensors på dette
        #Her kaldes en funktion der kører til første punkt i path med sensorer tændt mens den køre
        
    else: #roter så vi kan se current goal
        
    
    if #sensorer registrere noget inden for 40 cm, så stop kørsel, selflocalize og forfra
        
    
    if :#betingelse for at robot har været ved current goal
        current_goal+=1

