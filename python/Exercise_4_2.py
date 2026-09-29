import Exercise_4_2_rrt as rrt_class   
import Exercise_4_2_grid_robotmodel as grid_rm
import numpy as np
import matplotlib.pyplot  as plt

#import Exercise_1 as ex1
    
def robot_path(arlo,path):
    """Funktion til at køre den simpler path"""
    executed_path = [path[0]]
    
    path_length=len(path)
    
    for i in range(path_length - 1):

        x1, y1 = path[i] #from node
        x2, y2 = path[i + 1]# til node

        dx = x2 - x1 
        dy = y2 - y1

        distance = np.sqrt(dx**2 + dy**2)

        desired_angle = np.degrees(np.arctan2(dy, dx))
        
       # ex1.rotate_inplace(arlo,degrees=desired_angle)

        # Kør frem
        #ex1.straight_ahead(arlo,meters=distance)
        
        executed_path.append([x2, y2])

    return executed_path

        
    
import robot_models
import Exercise_4_1_plot as ex4_1
import json
#import robot

arlo_radius = 0.225 #robot.ARLO_RADIUS

#Hent info fra json
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


def main():
    
    START_POINT=[0, 0]
    GOAL=[0,5]
    GRID_CELL_SIZE= 0.1
    X_MIN=-2
    X_MAX=10
    Y_MIN=-2
    Y_MAX=10
    
    
    map = grid_rm.Grid(x_min=X_MIN,x_max=X_MAX,y_min=Y_MIN, y_max=Y_MAX, grid_cell_size=GRID_CELL_SIZE)
    map.obstacles(landmarks_list, robot_radius=arlo_radius) #generer vores landmarks som obstacles
    print()

    robot = grid_rm.PointMassModel(ctrl_range=[-GRID_CELL_SIZE, GRID_CELL_SIZE])   #

    rrt = rrt_class.RRT(start=START_POINT
              ,goal=GOAL,
        robot_model=robot,
        map=map,
        grid_cell_size=GRID_CELL_SIZE,
        )
    
    
    path = rrt.path_planning()
    
    
    if path is None:
            print("Cannot find path")
    else:
        print("found path!!")
        simpler_path=rrt.simpler_path(path)
        execute_path=robot_path(robot,simpler_path)
        
        
        fig, ax =plt.subplots()
        ex4_1.draw_landmarks(ax, landmarks_list, arlo_radius)
        
        
        executed_x = [p[0] for p in execute_path]
        executed_y = [p[1] for p in execute_path]

        ax.plot(executed_x,executed_y,'-',linewidth=2,label="Robot path")
        plt.grid(True)
        plt.pause(0.01)
        #ax.set_xlim(-10,10)
        #ax.set_ylim(-10,10)
        ax.scatter(GOAL[0],GOAL[1], color='green')
    
        plt.show()



if __name__ == '__main__':
    main()