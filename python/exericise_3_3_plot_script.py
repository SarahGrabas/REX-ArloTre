import matplotlib.pyplot as plt
import numpy as np

# Måling 1 - Data fra kørsel af exercise_3_3.py 
# ID 6: Xc = -0.518 m, Zc = 1.920 m
# ID 8: Xc = 0.233 m, Zc = 2.727 m
# ID 9: Xc = 0.667 m, Zc = 1.197 m

# Måling 2 - Data fra kørsel af exercise_3_3.py
# D 8: Xc = -0.040 m, Zc = 1.752 m
# ID 8: Xc = -0.214 m, Zc = 1.742 m
# ID 6: Xc = 0.181 m, Zc = 1.899 m
# ID 6: Xc = 1.049 m, Zc = 1.800 m

landmarks = [
    (8, (-0.040, 1.752)),
    (8, (-0.214, 1.742)),
    (6, (0.181, 1.899)),
    (6, (1.049, 1.800)),
]

for marker_id, coordinates in landmarks:
    print(f"Afstand til markør ID {marker_id}: {np.linalg.norm(coordinates):.3f} m")

fig, ax = plt.subplots()

ax.scatter(0, 0, color="black", label="Kamera")

for marker_id, (Xc, Zc) in landmarks:
    ax.scatter(Xc, Zc, color="tab:blue")
    ax.annotate(
        f"ID {marker_id}",
        (Xc, Zc),
        xytext=(5, 5),
        textcoords="offset points",
    )

ax.set_xlabel("Xc [m]")
ax.set_ylabel("Zc [m]")
ax.set_title("Markørernes positioner set ovenfra")
ax.set_aspect("equal", adjustable="box")
ax.margins(0.2)
ax.grid(True)
ax.legend()

fig.tight_layout()
fig.savefig("landmarkkort.png", dpi=150)
plt.show()
