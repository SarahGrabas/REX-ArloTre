from debug import debug
from rally import Robot, Landmark, ARLO_RADIUS, Camera
import numpy as np

# Looking for L1
## try to self localize
## 

## detect landmarks
###

def get_landmark(landmarks:list[Landmark], id:int) -> Landmark:
    """Assumes ``id`` is in ``landmarks``."""
    return [lm for lm in landmarks if lm.id == id][0]

def scan_for_landmark(arlo:Robot, id:int) -> bool:
    """Returns ``True`` iff can see ``id``."""
    angle = arlo.camera.FOV * 2/3
    total_angle = 0
    while total_angle < 360 - angle:
        landmarks = arlo.camera.detect_landmarks()



with Robot() as arlo:
    # FACT : Can see L1
    while True:
        l1 = get_landmark(arlo.camera.detect_landmarks(), 1)
        if l1.angle < 5: # degrees
            break
        arlo.motors.rotate(l1.angle)
    # FACT : Looking approximatly at L1
    landmarks = arlo.camera.detect_landmarks()
    goal = get_landmark(landmarks, 1).center
    goal -= (l1.radius + 0.4 + ARLO_RADIUS/2) * goal / np.linalg.norm # goal is the closest point on a circle around the L1
    # RRT PROTOCOL : OBSTACLES = landmarks, GOAL = gloal
    # FACT : Inside visit radius of L1
    total_angle = 0
    angle = Camera.FOV * 2/3
    while total_angle <= 360:
        total_angle += angle
        arlo.motors.rotate(angle)

