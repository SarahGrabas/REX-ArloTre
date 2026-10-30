import rally
import RRT
import selflocalize
import numpy as np
import time

arlo = rally.Robot()
arlo_radius =rally.ARLO_RADIUS *100 #fra meter til cm

safe_distance_to_lm = 0.4 #m, vi skal være tættere end dette på goal landmark

#Sonar tærskler. Sidesonarerne skal være lavere end RRT_BUFFER,
#ellers afbrydes stier, som RRT har planlagt tæt forbi en forhindring
safe_distance_front = 0.25 #m
safe_distance_sides = 0.15 #m
RRT_BUFFER = 20.0 #cm, afstand mellem robotkant og landmark-radius som RRT-stien holder

APPROACH_RANGE = 150.0 #cm, inden for denne afstand kører vi direkte til målet med kameraet
TARGET_GAP = 30.0 #cm, ønsket afstand fra robotfront til målets side ved ankomst

HORIZON_NODES = 3 #Antal noder i RRT path vi kører (dvs. 2 segmenter) før vi lokaliserer og planlægger igen
MAX_DRIVE_PER_CYCLE = 150.0 #cm, max distance vi kører før vi lokaliserer og planlægger igen
MAX_SCANS_BEFORE_MOVING = 8 #50 grader pr scan, dvs. ca. en hel omgang

valid_objects=[8,9,10,11]
valid_coords={
    8: (131, 131),
    9: (131,431 ), #466
    10: (531 , 131), #566
    11: (531 , 431)
}

phases=[8,9,10,11,8]
current_goal_index=0

#MAP
# X_MIN= 0.0
# X_MAX =  690
# Y_MIN= 0.0
# Y_MAX = 590

def motion_update(current_state):
    """Opdaterer partiklerne efter hver primitive bevægelse (én rotation eller én kørsel)"""
    selflocalize.motion_update(current_state, valid_objects, valid_coords)

def rotate(current_state, degrees):
    """Roterer robotten og opdaterer partiklerne med det samme"""
    arlo.motors.rotate(degrees)
    current_state["movement"][1]+=degrees
    motion_update(current_state)

def half_side(lm_id):
    """Halv sidelængde af landmark-kassen i cm (afstand fra centrum til siden)"""
    box_size = rally.Landmark.MAX_BOX_SIZES.get(lm_id, max(rally.Landmark.MAX_BOX_SIZES.values()))
    return box_size * 100 / 2

def sonars_choose_direction(current_state,arlo, distance, goal_coords=None):
    left = arlo.sonars.left()
    front = arlo.sonars.front()
    right = arlo.sonars.right()
    back = arlo.sonars.back()

    distances = {
        "left": left,
        "front": front,
        "right": right,
        "back": back
    }
    rotations = {"left": 90, "front": 0, "right": -90, "back": 180}

    print("Sonars:")
    print("  Left: ", left)
    print("  Front:", front)
    print("  Right:", right)
    print("  Back: ", back)

    # Retninger med nok plads til at køre distancen
    free = [d for d in distances if distances[d] * 100 > distance + safe_distance_front * 100]

    if goal_coords is not None and free:
        # Blandt de frie retninger vælger vi den der peger mest mod målet
        pose = current_state["est_pose"]
        goal_angle = np.degrees(np.arctan2(goal_coords[1] - pose.getY(), goal_coords[0] - pose.getX()))
        current_angle = np.degrees(pose.getTheta())

        def angle_to_goal(d):
            heading = current_angle + rotations[d]
            return abs((goal_angle - heading + 180) % 360 - 180)

        direction = min(free, key=angle_to_goal)
    else:
        # Find retningen med størst fri afstand
        direction = max(distances, key=distances.get)

    print("Choosing:", direction)

    # Drej mod den valgte retning (front betyder ingen rotation)
    if rotations[direction] != 0:
        rotate(current_state, rotations[direction])

    # Kør distance cm
    drive_with_sonars(current_state,arlo,distance, True)
    motion_update(current_state)

    return direction

def drive_with_sonars(current_state,arlo,cm, forward=True):

    meters = cm / 100

    full_route=True #Er robot blevet afbrudt undervejs

    # Kør ikke hvis der allerede er en forhindring foran
    if arlo.sonars.front() < safe_distance_front:
        print("Obstacle in front, not driving")
        return 0.0, False

    start = time.time()

    # Beregn hvor lang tid det burde tage
    drive_time = meters / arlo.motors.meters_pr_sec

    # Start motorerne uden at vente
    arlo.motors.drive(
        meters,
        forward,
        _sleep=False,
        stop_when_done=False
    )

    while True:
        current_time = time.time()

        left = arlo.sonars.left() < safe_distance_sides
        front = arlo.sonars.front() < safe_distance_front
        right = arlo.sonars.right() < safe_distance_sides

        # Stop hvis der er en forhindring
        if left or front or right:
            arlo.motors.stop()
            full_route=False
            break

        # Stop når den ønskede tid er gået
        if current_time - start >= drive_time:
            arlo.motors.stop()
            break

        time.sleep(0.01)

    total_driven_time = time.time() - start
    meter_driven = arlo.motors.meters_pr_sec * total_driven_time

    print("Centimeters driven:", meter_driven * 100)
    current_state["movement"][0]+=meter_driven * 100

    return meter_driven * 100, full_route

def execute_robot_path(current_state,arlo,path,number_nodes:int, max_distance:float):
    """Funktion til at køre den simpler path.
    Vi kører højst number_nodes noder og højst max_distance cm, før vi stopper og planlægger igen.
    Partiklerne opdateres efter hver rotation og hver kørsel."""

    executed_path = [path[0]]

    path_length=len(path)

    if number_nodes > path_length:
        number_nodes=path_length

    full_route=True
    remaining_distance = max_distance

    for i in range(number_nodes - 1): #Vi kører kun til antal noder angivet

        #Vi kører fra vores estimerede position, da den er opdateret efter sidste segment
        x1, y1 = current_state["est_pose"].getX(), current_state["est_pose"].getY()
        x2, y2 = path[i + 1]# til node

        dx = x2 - x1
        dy = y2 - y1

        distance = min(np.sqrt(dx**2 + dy**2), remaining_distance)

        #Robottens orientering
        current_angle = np.degrees(current_state["est_pose"].getTheta())
        desired_angle = np.degrees(np.arctan2(dy, dx)) % 360

        relative_angle = (desired_angle - current_angle + 180) % 360 - 180 #korteste rotationsvej
        print("desired_angle: ", desired_angle)
        print("relative_angle: ", relative_angle)
        print("distance: ", distance)

        rotate(current_state, relative_angle)
        driven, full_route=drive_with_sonars(current_state,arlo,distance) #input i cm
        motion_update(current_state)

        remaining_distance -= driven
        executed_path.append([x2, y2])

        if full_route==False or remaining_distance <= 0:
            break

    arlo.motors.stop()
    return executed_path, full_route

def rotate_to_goal(current_state, GOAL_COORDS):
    #Roter til goal ud fra estimeret pose
    current_x, current_y, current_theta=(current_state["est_pose"].getX(),current_state["est_pose"].getY(), current_state["est_pose"].getTheta())
    goal_x,goal_y =GOAL_COORDS

    dx = goal_x - current_x
    dy = goal_y - current_y
    current_angle =np.degrees(current_theta)

    # Vinkel fra robotten til målet
    desired_angle = np.degrees(np.arctan2(dy, dx)) % 360

    relative_angle = (desired_angle - current_angle + 180) % 360 - 180

    rotate(current_state, relative_angle)
    time.sleep(0.5)

def find_goal_in_frame(GOAL_ID):
    """Returnerer goal landmark hvis kameraet ser det, ellers None"""
    landmarks_seen = arlo.camera.detect_landmarks(unique=True)
    for lm in landmarks_seen:
        if lm.id == GOAL_ID:
            return lm
    return None

def goal_check(current_state,GOAL_ID, GOAL_COORDS):
    """Målet er nået hvis kameraet ser goal landmark, og både kamera og front sonar
    siger at vi er tættere end safe_distance_to_lm på kassens side."""
    rotate_to_goal(current_state, GOAL_COORDS)
    lm = find_goal_in_frame(GOAL_ID)

    if lm is None:
        print("goal not seen")
        return False

    #Drej så front sonaren peger direkte på målet
    rotate(current_state, lm.angle)
    time.sleep(0.3)

    camera_gap = lm.distance * 100 - half_side(GOAL_ID) - arlo_radius #cm fra robotfront til kassens side
    front = arlo.sonars.front()
    print(f"Goal check: camera gap {camera_gap:.1f} cm, front sonar {front*100:.1f} cm")

    #Front sonar skal måle målet (ikke en anden forhindring tættere på), så de to skal stemme nogenlunde overens
    return front < safe_distance_to_lm and camera_gap < safe_distance_to_lm * 100 + 10

def visual_approach(current_state, GOAL_ID):
    """Kører direkte mod målet med kameraets relative måling (visual servoing),
    hvis målet er synligt, tæt nok på og der ikke står noget imellem.
    Returnerer True hvis vi har kørt."""
    lm = find_goal_in_frame(GOAL_ID)
    if lm is None or lm.distance * 100 > APPROACH_RANGE:
        return False

    rotate(current_state, lm.angle)
    time.sleep(0.3)

    #Mål igen efter rotationen, nu skal målet være lige foran
    lm = find_goal_in_frame(GOAL_ID)
    if lm is None:
        return False

    gap = lm.distance * 100 - half_side(GOAL_ID) - arlo_radius #cm fra robotfront til kassens side
    front = arlo.sonars.front() * 100

    #Hvis sonaren måler noget væsentligt tættere end målet, står der en forhindring imellem
    if front < gap - 20:
        print("Something between robot and goal, using RRT instead")
        return False

    drive_distance = gap - TARGET_GAP
    print(f"Visual approach: gap {gap:.1f} cm, driving {drive_distance:.1f} cm")
    if drive_distance > 0:
        drive_with_sonars(current_state, arlo, drive_distance)
        motion_update(current_state)
    return True

def localize(current_state, min_landmarks):
    """Scanner efter landmarks indtil partiklerne er samlet (lav spredning)
    og vi har set mindst min_landmarks forskellige valid landmarks."""
    valid_lm_seen = []
    scan_counter = 1
    while True:
        if scan_counter % MAX_SCANS_BEFORE_MOVING == 0: #Hvis vi har drejet en hel omgang uden at være sikre, bevæger vi os lidt
            sonars_choose_direction(current_state, arlo, 50)

        if scan_counter != 1:
            selflocalize.scan_function(current_state, arlo)

        print("Selflocalizing")
        landmarks_seen = arlo.camera.detect_landmarks(unique=True) #Hvis vi ser flere med samme ID, tager vi gennemsnit

        all_obj_in_frame, valid_lm_seen = selflocalize.data_objects(landmarks_seen, valid_lm_seen, valid_objects)
        selflocalize.mcl_update(current_state, all_obj_in_frame, valid_objects, valid_coords)

        if len(valid_lm_seen) >= min_landmarks and selflocalize.is_converged(current_state):
            return

        scan_counter += 1

def scan_obstacles(current_state, GOAL_ID):
    """Scanner ligeud, 50 grader til venstre og 50 grader til højre og samler forhindringer i globale koordinater.
    De kendte landmarks (undtagen målet) er altid forhindringer."""
    total_obstacles = {}

    for lm_id, coords in valid_coords.items():
        if lm_id != GOAL_ID:
            total_obstacles[lm_id] = (lm_id, coords, half_side(lm_id) * np.sqrt(2))

    for rotation in (0, 50, -100):
        if rotation != 0:
            rotate(current_state, rotation)

        landmarks_seen = arlo.camera.detect_landmarks(unique=True)
        pose = (current_state["est_pose"].getX(), current_state["est_pose"].getY(),current_state["est_pose"].getTheta())
        obstacles = RRT.convert_lm_center_to_global(pose,landmarks_seen,GOAL_ID)

        for obstacle in obstacles:
            if obstacle[0] not in total_obstacles:
                total_obstacles[obstacle[0]] = obstacle

    return tuple(total_obstacles.values())

def stop_point(current_state, GOAL_ID, GOAL_COORDS):
    """Punktet foran målet (på linjen mod robotten), hvor robottens centrum skal ende"""
    x, y = current_state["est_pose"].getX(), current_state["est_pose"].getY()
    goal_x, goal_y = GOAL_COORDS
    dx, dy = x - goal_x, y - goal_y
    distance = np.sqrt(dx**2 + dy**2)
    offset = half_side(GOAL_ID) + arlo_radius + TARGET_GAP

    if distance <= offset:
        return (x, y)
    return (goal_x + dx / distance * offset, goal_y + dy / distance * offset)

#Kalibrering kræver et synligt landmark, ellers kører vi med standard værdierne
try:
    arlo.calibrate_velocities()
except AssertionError as e:
    print("Calibration skipped:", e)

current_state=selflocalize.initialize_mcl(num_particles=1500)
first_localization = True

while current_goal_index < len(phases):

    GOAL_ID=phases[current_goal_index]
    GOAL_COORDS=valid_coords[GOAL_ID]
    print(f"Current goal: {GOAL_ID}")

    #Selflocalization. Første gang skal vi se 2 landmarks, bagefter har partiklerne en god prior
    localize(current_state, min_landmarks=2 if first_localization else 1)
    first_localization = False

    rotate_to_goal(current_state, GOAL_COORDS)
    print("Rotating to Goal")

    #Er målet tæt på og frit, kører vi direkte med kameraet
    if not visual_approach(current_state, GOAL_ID):
        obstacles = scan_obstacles(current_state, GOAL_ID)
        rotate_to_goal(current_state, GOAL_COORDS)

        print(f"RRT TO ID: {GOAL_ID}")
        path=RRT.RRT_call(stop_point(current_state, GOAL_ID, GOAL_COORDS), obstacles , current_state["est_pose"], arlo_radius, buffer=RRT_BUFFER)

        if path: #ikke er none
            print("Path found: ", path)
            #Vi kører kun et stykke af pathen, og lokaliserer og planlægger igen i næste iteration
            _, full_route=execute_robot_path(current_state,arlo,path,HORIZON_NODES, MAX_DRIVE_PER_CYCLE)

            if full_route==False:
                sonars_choose_direction(current_state,arlo,35, GOAL_COORDS)
        else:
            print(f"Could not find path to {GOAL_ID}")
            #Vi kører lidt væk fra obstacles, så vi kan scanne igen
            sonars_choose_direction(current_state,arlo,50, GOAL_COORDS)

    if goal_check(current_state, GOAL_ID, GOAL_COORDS):
        print(f"Goal {GOAL_ID} reached")
        current_goal_index += 1

print("Rally finished")
arlo.motors.stop()
