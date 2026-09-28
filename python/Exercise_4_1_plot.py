import matplotlib.pyplot as plt
import numpy as np
import json

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
    
       
for marker_id, centrum, radius in landmarks_list:
    print(f"Afstand til markør ID {marker_id}: {np.linalg.norm(tuple(centrum))+ radius:.3f} m")

fig, ax = plt.subplots()

ax.scatter(0, 0, color="black", label="Kamera")

for marker_id, centrum, radius in landmarks_list:
    X=centrum[0]
    Y=centrum[1]
    ax.scatter(X, Y, color="tab:blue")
    ax.annotate(
        f"ID {marker_id}",
        (X, Y),
        xytext=(5, 5),
        textcoords="offset points",
    )
    circle = plt.Circle((X, Y), radius, fill=False)
    ax.add_patch(circle)

ax.set_xlabel("X")
ax.set_ylabel("Y (forward)")
ax.set_title("Markørernes positioner set ovenfra med robot i origo")
ax.set_aspect("equal", adjustable="box")
#ax.margins(0.2)
ax.grid(True)
ax.legend()

fig.tight_layout()
#fig.savefig("landmarkkort.png", dpi=150)
plt.show()

