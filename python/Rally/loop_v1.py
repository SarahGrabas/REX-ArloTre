from debug import debug
from rally import Robot, Landmark, ARLO_RADIUS, Camera
import numpy as np
from loop_v1_rrt import Grid, RRT
from time import perf_counter
import random



def get_landmark(landmarks:list[Landmark], id:int) -> Landmark:
    try:
        return [lm for lm in landmarks if lm.id == id][0]
    except IndexError:
        None

def scan_for_landmark(arlo:Robot, id:int) -> Landmark:
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

def look_at_landmark(arlo:Robot, id:int) -> Landmark:
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

def RRT_instructions(goal:list, obstacles:tuple[Landmark, ...]):

    START_POINT=[0, 0]
    GOAL=goal

    GRID_CELL_SIZE= 0.1
    X_MIN=-20
    X_MAX=20
    Y_MIN=-20
    Y_MAX=20
    
    map = Grid(X_MIN, X_MAX, Y_MIN, Y_MAX, GRID_CELL_SIZE)
    map.obstacles([(lm.id, lm.center, lm.radius) for lm in obstacles], robot_radius=ARLO_RADIUS) #generer vores landmarks som obstacles
    
    rrt = RRT(start=START_POINT,goal=GOAL,map=map,grid_cell_size=GRID_CELL_SIZE,)
        
    path = rrt.path_planning()
    
    if path is None:
        print("Cannot find path")
        return None
    else:
        simple_path = rrt.simpler_path(path)
        instructions = rrt.generate_instructions(simple_path)
        return instructions

def drive_with_sensors(arlo:Robot, meters:float):
    BUFFER = 0.1

    obstacles = arlo.camera.detect_landmarks()
    a = tuple(obs for obs in obstacles if obs.center[1] <= meters + obs.radius + ARLO_RADIUS + BUFFER)
    b = tuple(obs for obs in a if abs(obs.center[0]) <= ARLO_RADIUS + BUFFER)

    if b:
        m = min(obs.center[1] for obs in b)
        meters = min(m, meters)

    if min(arlo.sonars.left(), arlo.sonars.front(), arlo.sonars.right()) >= 0.2:
        time = arlo.motors.drive(meters, True, False, False)
        START_TIME = perf_counter()
        while (perf_counter()-START_TIME)*1_000_000 < time:
            if min(arlo.sonars.left(), arlo.sonars.front(), arlo.sonars.right()) < 0.2:
                arlo.motors.stop()
                raise AssertionError()
        arlo.motors.stop()
    else:
        raise AssertionError()




with Robot() as arlo:
    for target_id in (1, 2, 3, 4, 1):
        while True:
            target = look_at_landmark(arlo, target_id)
            avoid = {i for i in (1, 2, 3, 4, 1) if i != target_id}
            if target is None:
                arlo.motors.rotate(random.randint(-180, 180))
                drive_with_sensors(arlo, 1)
            else:
                obstacles = arlo.camera.detect_landmarks()
                goal = target.center[:2]
                goal -= (target.radius + 0.4 + ARLO_RADIUS/2) * goal / np.linalg.norm(goal) # goal is the closest point on a circle around the L1
                instructions = None
                while instructions is None:
                    instructions = RRT_instructions(goal, obstacles)
                for action,param in instructions:
                    try:
                        if action == "rotate":
                            arlo.motors.rotate(param)
                        elif action == "drive":
                            drive_with_sensors(arlo, param)
                        else:
                            raise TypeError()
                    except AssertionError:
                        break
                break

