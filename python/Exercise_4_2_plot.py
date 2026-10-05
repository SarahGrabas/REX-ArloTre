import json
import Exercise_4_1 as ex4_1
import robot
import matplotlib.pyplot as plt


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


path =[]
execute_path=[]
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