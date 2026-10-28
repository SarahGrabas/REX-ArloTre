import Exercise_4_2_rrt as rrt_class   
import Exercise_4_2_grid as grid
import numpy as np
from typing import Sequence, Union
from robot import Robot, sleep, arlo


def robot_path(
    arlo: Robot,
    path: "Sequence[Union[Sequence[float], np.ndarray]]",
) -> "list[Union[Sequence[float], np.ndarray]]":
    """Funktion til at køre den simpler path
    Gridsize er længde på x og y akse i meter"""

    executed_path = [path[0]]
    
    path_length=len(path)

    current_angle = 0  # Robotten starter med at pege ligeud
    
    for i in range(path_length - 1):

        x1, y1 = path[i] #from node
        x2, y2 = path[i + 1]# til node

        dx = x2 - x1
        dy = y2 - y1

        distance = np.sqrt(dx**2 + dy**2)

        desired_angle = -np.degrees(np.arctan2(dx, dy))

        relative_angle = desired_angle - current_angle
        
        print("         desired_angle, relative_angle, distance",
              desired_angle, relative_angle, distance)

        arlo.rotate(relative_angle, stop_when_done=False)
        arlo.drive(distance, stop_when_done=False)

        current_angle = desired_angle
        
        executed_path.append([x2, y2])

    arlo.stop()
    return executed_path




import json
import robot
arlo_radius=robot.ARLO_RADIUS

#Hent info fra json
try:
    with open('coordinates.json') as file:
        data = json.load(file)
        landmarks=data
        landmarks_list=[]
        for landmark in landmarks:
            id=landmark["id"]
            center=landmark["center"]
            radius=landmark["radius"]
            landmarks_list.append((id,center,radius))
        print(landmarks_list)
except:
    with open('./python/coordinates.json') as file:
        data = json.load(file)
        landmarks=data
        landmarks_list=[]
        for landmark in landmarks:
            id=landmark["id"]
            center=landmark["center"]
            radius=landmark["radius"]
            landmarks_list.append((id,center,radius))
        print(landmarks_list)



if __name__ == '__main__':

    START_POINT=[0, 0]
    GOAL=[0, 3.61]

    GRID_CELL_SIZE= 0.1
    X_MIN=-0.7
    X_MAX=0.7
    Y_MIN=0
    Y_MAX=4
    
    
    map = grid.Grid(X_MIN, X_MAX, Y_MIN, Y_MAX, GRID_CELL_SIZE)
    map.obstacles(landmarks_list, robot_radius=arlo_radius) #generer vores landmarks som obstacles
    
    rrt = rrt_class.RRT(start=START_POINT,goal=GOAL,map=map,grid_cell_size=GRID_CELL_SIZE,)
    
    
    path = rrt.path_planning()
    
    if path is None:
            print("Cannot find path")
    else:
        print("found path!!")
        print(path)
        simpler_path=rrt.simpler_path(path)
        print(simpler_path)
        execute_path=robot_path(arlo,simpler_path)
        # execute_path=robot_path(arlo,path)
        print(execute_path)

