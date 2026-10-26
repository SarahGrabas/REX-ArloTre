from debug import debug
from rally import Robot, Landmark, ARLO_RADIUS, Camera
import numpy as np



def get_landmark(landmarks:list[Landmark], id:int) -> Landmark|None:
    try:
        return [lm for lm in landmarks if lm.id == id][0]
    except IndexError:
        None

def scan_for_landmark(arlo:Robot, id:int) -> Landmark|None:
    """Returns the landmark iff can see landmark with ``id``."""
    angle = arlo.camera.FOV * 2/3
    total_angle = 0
    while total_angle <= 360:
        lm = get_landmark(arlo.camera.detect_landmarks(True), id)
        if lm is not None:
            return lm
        else:
            if total_angle <= 360 - angle:
                total_angle += angle
                arlo.motors.rotate(angle)
            else:
                arlo.motors.rotate(360 - angle)
                break
    lm = get_landmark(arlo.camera.detect_landmarks(True), id)
    if lm is not None:
        return lm
    return None

def look_at_landmark(arlo:Robot, id:int) -> Landmark|None:
    """Returns the landmark iff ``arlo`` is looking approximatly at landmark with ``id`` (returns ``None`` if cannot find landmark with ``id``)."""
    while True:
        lm = scan_for_landmark(arlo, id)
        if lm is None: # if we could not find it, return None
            return None
        else: 
            if lm.angle < 7: #degrees
                return lm
            else: # if we could find it, but not looking directly at it
                arlo.motors.rotate(lm.angle)



with Robot() as arlo:
    for target_id in (1, 2, 3, 4, 1):
        while True:
            target = look_at_landmark(arlo, target_id)
            avoid = {i for i in (1, 2, 3, 4, 1) if i != target_id}
            if target is None:
                # GO RANDOM DIRECTION
                # AVOID OBSTACLES
                # AVOID OTHER TARGETS
                pass
            else:
                obstacles = arlo.camera.detect_landmarks(True)
                goal = target.center
                goal -= (target.radius + 0.4 + ARLO_RADIUS/2) * goal / np.linalg.norm # goal is the closest point on a circle around the L1
                # USE RRT TO PLAN PATH TO GOAL
                # DRIVE PATH
                # AVOID OBSTACLES
                # AVOID OTHER TARGETS
                break

