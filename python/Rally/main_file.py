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
    landmarks_seen = arlo.camera.detect_landmarks()

    valid_objects =[] #dem vi vil selflocalize ud fra
    
    all_obj_in_frame, valid_coords=selflocalize.sorting_objects(valid_objects,landmarks_seen)
    selflocalize.mcl_update(current_state,all_obj_in_frame,valid_objects,valid_coords)

    selflocalize.scan_function(current_state,arlo)
    
    if phases[current_goal] in all_obj_in_frame: #hvis vi kan se current goal, så laver vi RRT til den
        GOAL_ID=phases[current_goal]
        
        GOAL,obstacles=RRT.convert_lm_center_to_global(landmarks_seen, GOAL_ID)
        path=RRT.RRT_call(GOAL, obstacles , current_state["est_pose"], arlo_radius)
        
        number_nodes=len(path) #antal noder fra path vi ønsker at kører til
        RRT.execute_robot_path(arlo,path,number_nodes)
        #Vi mangler at putte sensors på dette
        #Her kaldes en funktion der kører til første punkt i path med sensorer tændt mens den køre
        
        if #sensorer registrere noget inden for 40 cm, så stop kørsel, selflocalize og forfra
        
    
    if :#betingelse for at robot har været ved current goal
        current_goal+=1

