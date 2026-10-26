import Exercise_4_2_rrt as rrt_class   
import Exercise_4_2_grid as grid
import numpy as np
from typing import Sequence, Union
from robot import Robot, sleep, arlo
# import matplotlib.pyplot as plt

import Exercise_1 as ex1
import json

import robot
import rally_copy

robot_controller = rally_copy.Robot()

arlo_radius=rally_copy.ARLO_RADIUS

def RRT_call(GOAL:tuple[float, float],obstacles: tuple[int,tuple[float,float],float], estimate_robot_pose:tuple[float,float,float]):
    X_MIN= 0.0
    X_MAX=120+22+400+28+120
    Y_MIN= 0.0
    Y_MAX= 120+22+300+28+120
    GRID_CELL_SIZE=0.1
    map = grid.Grid(X_MIN,X_MAX,Y_MIN,Y_MAX, GRID_CELL_SIZE)
    map.obstacles(obstacles, robot_radius=arlo_radius)
    
    START_POINT=[estimate_robot_pose[0],estimate_robot_pose[1]]
    rrt = rrt_class.RRT(start=START_POINT,goal=GOAL,map=map,grid_cell_size=GRID_CELL_SIZE,)
    path = rrt.path_planning()
    
    if path is None:
            print("Cannot find path")
    else:
        print("found path!!")
        print(path)
        simpler_path=rrt.simpler_path(path)
        print(simpler_path)
    
    
    return simpler_path

obstacles = robot_controller.camera.detect_landmarks()

detected_objectIDs = [lm.id for lm in obstacles]
detected_center = [lm.center  for lm in obstacles]
detected_radius = [lm.radius for lm in obstacles]

detected_obj =[]
for i in range(len(detected_objectIDs)):
    if detected_objectIDs[i] is not 1:
        detected_obj.append(detected_objectIDs[i], detected_center[i], detected_radius[i])
    
    if detected_objectIDs[i]== 1:
        GOAL = (detected_center[i])


print(detected_obj)
RRT_call(GOAL, detected_obj, )
    

# execute_path=robot_path(arlo,simpler_path)
# print(execute_path)