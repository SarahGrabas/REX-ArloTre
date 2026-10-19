import json
import Exercise_4_1_plot as ex4_1
import matplotlib.pyplot as plt
import numpy as np


arlo_radius=0.225

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


path = [[0, 0], np.array([-0.05841044,  0.4965765 ]), np.array([-0.07257802,  0.99637574]), np.array([-0.10749262,  1.49515522]), np.array([-0.59065951,  1.36650938]), np.array([-0.62683552,  1.86519896]), np.array([-0.63725333,  2.36509042]), np.array([-0.63326477,  2.86507451]), np.array([-0.38856462,  3.30110414]), np.array([-0.06011646,  3.2923513 ]), np.array([-0.1013363 ,  3.79064933]), np.array([0.13382039, 4.23189906]), np.array([0.15628073, 4.73139434]), np.array([0.28105837, 5.21557462]), np.array([-0.08225367,  5.55909225]), np.array([0.  , 5.89])]

execute_path=[[0, 0], [-0.6268355178567913, 1.8651989630344847], [-0.6332647742985531, 2.8650745114725087], [0.0, 5.89]]


GOAL=[0, 5.89]
        
        
fig, ax =plt.subplots()
ex4_1.draw_landmarks(ax, landmarks_list, arlo_radius)
        
pathx = [p[0] for p in path]
pathy = [p[1] for p in path]

ax.plot(pathx,pathy,'-',linewidth=2,label="RRT path")
        
print(execute_path)
executed_x = [p[0] for p in execute_path]
executed_y = [p[1] for p in execute_path]

ax.plot(executed_x,executed_y,'-',linewidth=2,label="Robot path")
plt.grid(True)
plt.pause(0.01)
ax.scatter(GOAL[0],GOAL[1], color='green')
    
plt.show()