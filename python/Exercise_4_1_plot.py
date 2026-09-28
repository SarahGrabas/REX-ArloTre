import matplotlib.pyplot as plt
import numpy as np
import json
#import robot

arlo_radius =  0.225 #robot.ARLO_RADIUS

with open('/Users/sarahgrabas/Desktop/REX/REX-ArloTre/python/coordinates.json') as file:
    data = json.load(file)
    print("Type:", type(data))
    landmarks=data
    landmarks_list=[]
    for landmark in landmarks:
        id=landmark["id"]
        center=landmark["center"]
        radius=landmark["radius"]
        landmarks_list.append((id,center,radius))
    print(landmarks_list)
    
fig, ax = plt.subplots()
       
for marker_id, centrum, radius_box in landmarks_list:
    X=centrum[0]
    Y=centrum[1]
    
    distance=np.linalg.norm(tuple(centrum))- radius_box-arlo_radius
    print(f"Afstand til markør ID {marker_id}: {distance:.3f} m")
    
    circle_box = plt.Circle((X, Y), radius_box, edgecolor='red', facecolor='lightblue', linewidth=1, fill=True)
    ax.add_patch(circle_box)
    ax.scatter(X,Y, color="red")
    
circle_robot= plt.Circle((0, 0), arlo_radius, linewidth=2, edgecolor='black', facecolor='grey', fill=True)
ax.add_patch(circle_robot)
ax.scatter(0, 0, color="black", label="Robot") #Robot i origi
    
    

ax.set_xlabel("X")
ax.set_ylabel("Y (forward)")
ax.set_title("Markørernes positioner set ovenfra med robot i origo")
ax.set_aspect("equal", adjustable="box")
ax.autoscale()
#ax.margins(0.2)
ax.grid(True)
ax.legend()

fig.tight_layout()
#fig.savefig("landmarkkort.png", dpi=150)
plt.show()

