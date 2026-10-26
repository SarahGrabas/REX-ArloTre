import Exercise_4_2_rrt as rrt_class   
import Exercise_4_2_grid as grid
import numpy as np
from typing import Sequence, Union
from robot import Robot, sleep, arlo
# import matplotlib.pyplot as plt

import Exercise_1 as ex1
    
# def robot_path(arlo:Robot,path):
#     """Funktion til at køre den simpler path
#     Gridsize er længde på x og y akse i meter"""
    
#     executed_path = [path[0]]
    
#     path_length=len(path)
    
#     for i in range(path_length - 1):

#         x1, y1 = path[i] #from node
#         x2, y2 = path[i + 1]# til node

#         dx = x2 - x1
#         dy = y2 - y1

#         distance = np.sqrt(dx**2 + dy**2)

#         desired_angle = -np.degrees(np.arctan2(dx, dy))
        
#         print("         desired_angle, distance", desired_angle, distance)
#         arlo.rotate(desired_angle)
#         sleep(0.5)
#         arlo.drive(distance)
#         sleep(0.5)
        
#         executed_path.append([x2, y2])

#     return executed_path


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


        
    
#import python.OLD.robot_models as robot_models
#import Exercise_4_1_plot as ex4_1
import json

import robot


arlo_radius=robot.ARLO_RADIUS

# #Hent info fra json
# try:
#     with open('coordinates.json') as file:
#         data = json.load(file)
#         landmarks=data
#         landmarks_list=[]
#         for landmark in landmarks:
#             id=landmark["id"]
#             center=landmark["center"]
#             radius=landmark["radius"]
#             landmarks_list.append((id,center,radius))
#         print(landmarks_list)
# except:
#     with open('./python/coordinates.json') as file:
#         data = json.load(file)
#         landmarks=data
#         landmarks_list=[]
#         for landmark in landmarks:
#             id=landmark["id"]
#             center=landmark["center"]
#             radius=landmark["radius"]
#             landmarks_list.append((id,center,radius))
#         print(landmarks_list)


def RRT_call(GOAL:tuple[float, float],obstacles: tuple[int,tuple[float,float],float], estimate_robot_pose:tuple[float,float,float]):
    X_MIN= 0.0
    X_MAX=
    Y_MIN=
    Y_MAX=
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

landmarks_seen = arlo.camera.detect_landmarks()

    detected_objectIDs = [lm.id for lm in landmarks_seen]
    detected_dists = [lm.distance * 100.0 for lm in landmarks_seen]

    detected_angles = [np.deg2rad(lm.angle) for lm in landmarks_seen]

    detected_obj =(detected_objectIDs, detected_dists, detected_angles)
    print(detected_obj)
RRT_call()
    

# execute_path=robot_path(arlo,simpler_path)
# print(execute_path)


