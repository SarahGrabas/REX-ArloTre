import Exercise_4_2_rrt as rrt_class   
import Exercise_4_2_grid_robotmodel as grid_rm
import numpy as np
# import matplotlib.pyplot as plt

import Exercise_1 as ex1
    
def robot_path(arlo,path):
    """Funktion til at køre den simpler path
    Gridsize er længde på x og y akse i meter"""
    
    executed_path = [path[0]]
    
    path_length=len(path)
    
    for i in range(path_length - 1):

        x1, y1 = path[i] #from node
        x2, y2 = path[i + 1]# til node

        dx = x2 - x1
        dy = y2 - y1

        distance = np.sqrt(dx**2 + dy**2)

        desired_angle = np.degrees(np.arctan2(dy, dx))
        

        ex1.rotate_inplace(arlo,degrees=desired_angle)

        ex1.straight_ahead(arlo,meters=distance)
        
        executed_path.append([x2, y2])

    return executed_path

        
    
import robot_models
#import Exercise_4_1_plot as ex4_1
import json

from robot import Robot
import robot

arlo = Robot 

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
    GOAL=[-1.5,2.5]

    GRID_CELL_SIZE= 0.1
    X_MIN=-2
    X_MAX=1
    Y_MIN=0
    Y_MAX=3
    
    
    map = grid_rm.Grid(X_MIN, X_MAX, Y_MIN, Y_MAX, GRID_CELL_SIZE)
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
        
        
        # fig, ax =plt.subplots()
        # ex4_1.draw_landmarks(ax, landmarks_list, arlo_radius)
        
        # pathx = [p[0] for p in path]
        # pathy = [p[1] for p in path]

        # ax.plot(pathx,pathy,'-',linewidth=2,label="RRT path")
        
        # print(path)
        # print(execute_path)
        # executed_x = [p[0] for p in execute_path]
        # executed_y = [p[1] for p in execute_path]

        # ax.plot(executed_x,executed_y,'-',linewidth=2,label="Robot path")
        # plt.grid(True)
        # plt.pause(0.01)
        # ax.scatter(GOAL[0],GOAL[1], color='green')
    
        # plt.show()
    