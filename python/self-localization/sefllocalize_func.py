import cv2
import particle
#import camera
import numpy as np
from timeit import default_timer as timer
import sys
from math import pi as PI
import time
#import robot
#import robot as robot_module
from selflocalize_s import draw_world
import random
import random_numbers as rn
import rally

WHEEL_2_WHEEL_DIST= 0.381

meters_pr_sec = 2 * PI * WHEEL_2_WHEEL_DIST / 8 * (3.14/3)
degrees_pr_second = 360 / 4

#V_CALIB = 40.0   # Den kører 40 cm pr. sekund ca. Fremadrettet hastighed ved go_diff_calibrated(1, 1)
#W_CALIB = 1.9234  # I radianer/pr. sekund. Rotationshastighed (~110,2 degree/s) ved go_diff_calibrated(-1, 1)

V_CALIB = meters_pr_sec * 100
W_CALIB = np.deg2rad(degrees_pr_second)

SCAN_PAUSE_TIME = 0.40

robot_controller = rally.Robot()

# Landmarks.
# The robot knows the position of the 4 landmarks. Their coordinates are in the unit centimeters [cm].
landmarkIDs = [7,6]
#Coordinates for each landmark
landmarks = {
    #2: (0.0, 0.0),
    #4: (100.0, 0.0),
    7: (130.0,250.0),
    6: (0.0,250.0)
    
}

#Motion model, opdatere partikler ud fra motion kun
def sample_motion_model_velocity(particles, velocity, angular_velocity, delta_t):

    # a1: Hvor meget usikkerhed der kommer på translationshastigheden,
    #     når robotten kører fremad. Større a1 betyder større variation
    #     i hvor langt hver particle bevæger sig.
    #
    # a2: Hvor meget robotrotation bidrager til usikkerhed i
    #     translationshastigheden. Hvis robotten både kører og drejer,
    #     kan rotationen derfor give ekstra usikkerhed i positionen.
    #
    # a3: Hvor meget robotternes translation bidrager til usikkerhed
    #     i rotationshastigheden. Større a3 giver mere variation
    #     i hvor meget robotten drejer, når den kører.
    #
    # a4: Hvor meget selve rotationen bidrager til usikkerhed i
    #     rotationshastigheden. Større a4 betyder større variation
    #     i den estimerede drejevinkel ved rotation.
    #
    # a5: Ekstra støj på den endelige orientering, som afhænger af
    #     robotternes translationshastighed. Bruges til at modellere,
    #     at kørsel også kan påvirke robotens orientering.
    #
    # a6: Ekstra støj på den endelige orientering, som afhænger af
    #     robotternes rotationshastighed. Større a6 giver mere usikkerhed
    #     i orienteringen efter en rotation.
    
    a1 = 0.40
    a2 = 0.20
    a3 = 0.20
    a4 = 0.20
    a5 = 0.10
    a6 = 0.10

    for p in particles:

        # Hvis robotten roterer på stedet, skal translationshastigheden være 0
        if abs(velocity) < 1e-6:
            v_hat = 0.0

        else:

            v_hat = velocity + np.random.normal(0,np.sqrt(a1 * velocity**2 +a2 * angular_velocity**2))

        w_hat = angular_velocity + np.random.normal(0,np.sqrt(a3 * velocity**2 +a4 * angular_velocity**2))

        gamma_hat = np.random.normal(0,np.sqrt(a5 * velocity**2 +a6 * angular_velocity**2))

        x = p.getX()
        y = p.getY()
        theta = p.getTheta()

        x_new = x + v_hat * delta_t * np.cos(theta)
        y_new = y + v_hat * delta_t * np.sin(theta)

        theta_new = np.mod(theta + w_hat * delta_t + gamma_hat * delta_t,2.0 * np.pi)

        p.setX(x_new)
        p.setY(y_new)
        p.setTheta(theta_new)

def initialize_particles(num_particles):
    particles = []
    for i in range(num_particles):
        # Random starting points. 
        p = particle.Particle(600.0*np.random.ranf() - 100.0, 600.0*np.random.ranf() - 250.0, np.mod(2.0*np.pi*np.random.ranf(), 2.0*np.pi), 1.0/num_particles)
        particles.append(p)

    return particles

#This is our observation model
def update_particle_weights(particles, objectIDs, dists, angles,landmarks):
    #Vi beregn weight ved distance og vinkler
    particle_weights =[]
    #Hvis particle filtering ikke konvergerer kan vi ændre på sigma
    sigma_dist = 15.0 #cm
    sigma_angle=0.10
    print(objectIDs, dists)
    
    for p in particles:
        x = p.getX()
        y = p.getY()
        theta = p.getTheta()
        weight = 1.0 

        for i in range(len(objectIDs)):
        #Landmarkets coordinater 
            x_obj, y_obj = landmarks[objectIDs[i]]
            
        #Beregn euclidean distance fra partikel til landmark
            diff_x = x_obj - x
            diff_y = y_obj - y
            particle_dist = np.sqrt(diff_x**2 + diff_y**2)
            
        #Kameraets målte afstand til landmark
            measured_dist = dists[i]

        #Gaussian probability
        #(particle_dist - measured_dist) er forskellen mellem opringelig afstand og målte afstand hvis robotten er på denne partikle position.
            land_dist = (1 / (np.sqrt(2 * np.pi) * sigma_dist)) * np.exp(-0.5 * ((particle_dist - measured_dist) / sigma_dist)**2)
            
        #Vinkel beregninger
            enheds_theta=np.array([np.cos(theta), np.sin(theta)])
            enheds_particle=np.array([diff_x, diff_y])/np.linalg.norm(np.array([diff_x, diff_y]))
            
            dot_product = np.dot(enheds_theta, enheds_particle)
            dot_product = np.clip(dot_product, -1.0, 1.0) #Sikrer at vi ikke får tal som 1.00000002

            theta_particle = np.arccos(dot_product)     
            
            #Vi skal finde ud af om theta_p er negativ eller positiv, dvs om den ligger på højre eller venstre side af enheds theta
            #enheds_theta_hat=np.array(np.sin(theta),-np.cos(theta)) #Denne vektor ville vi skulle bruge hvis kamrea har positiv vinkel til højre
            enheds_theta_hat=np.array([-np.sin(theta),np.cos(theta)])   #Denne vektor bruger vi da kamrea har positiv vinkel til venstre
            
            final_theta_particle=np.sign(np.dot(enheds_particle, enheds_theta_hat))*theta_particle
            
            #Gaussian probability
            #(final_theta_particle - angles) er forskellen mellem opringelig vinkel og målte vinkel hvis robotten har denne partikel vinkel.
            land_angle = (1 / (np.sqrt(2 * np.pi) * sigma_angle)) * np.exp(-0.5 * ((final_theta_particle - angles[i]) / sigma_angle)**2)
            
            land_weight=land_dist*land_angle
            
            #Hvis der er flere landmarks, ganges deres sandsynligheder sammen
            weight *= land_weight 
        
        particle_weights.append(weight)    
        
    particle_weights = np.array(particle_weights,dtype=float)

    total_weight = np.sum(particle_weights)

    #Vi normaliserer vægtene
    if total_weight > 0:
            particle_weights = particle_weights / total_weight
    else:
            particle_weights[:] = 1.0 / len(particle_weights)
        
    for p, p_weight in zip(particles,particle_weights):
            p.setWeight(p_weight)
            
def resample_particles(particles):

    num_particles = len(particles)
    num_random = int(0.10 * num_particles)
    num_resampled = num_particles - num_random

    weights = np.asarray(
        [p.getWeight() for p in particles],
        dtype=float
    )

    total = np.sum(weights)

    if total <= 0 or not np.isfinite(total):
        weights = np.ones(num_particles) / num_particles
    else:
        weights = weights / total

    # Vælg 90 % af partiklerne ud fra deres vægt
    selected = np.random.choice(
        num_particles,
        size=num_resampled,
        replace=True,
        p=weights
    )
    print("RESAMPLING DEBUG:")
    print("most selected index:", np.bincount(selected).argmax())
    print("most selected count:", np.bincount(selected).max())

    best_index = np.argmax(weights)
    print("original best index:", best_index)
    print("original best weight:", weights[best_index])
    print("best selected count:", np.sum(selected == best_index))

    new_particles = []

    # De 90 % resamples fra de eksisterende partikler
    for index in selected:
        new_particle = particle.Particle(
            particles[index].getX(),
            particles[index].getY(),
            particles[index].getTheta(),
        )
        new_particles.append(new_particle)

    # De sidste 10 % bliver helt nye random partikler
    random_particles = initialize_particles(num_random)
    new_particles.extend(random_particles)

    for p in new_particles:
        p.setWeight(1.0 / num_particles)
        
    particle.add_uncertainty(new_particles, 3.0, 0.05)

    return new_particles

    

def weight_based_cam(particles, objectIDs, dists, angles,landmarks):
    """Funktion at opdaterer weights baseret på objekts og resample."""
    if objectIDs:
        update_particle_weights(
            particles,
            objectIDs,
            dists,
            angles,landmarks
        )
    
        particles = resample_particles(particles)
        
    return particles
    
    
def mcl_step(state, u_t, z_t, delta_t,landmarks):
    """
    Funktion der udfører MCL-skridt: Prediction, Correction og Resampling.
    Vi beregner particles movement ud fra velocity og angular verlocity som burde afspejle robottens bevægelse.
    Vi opdaterer partikel vægt og resampler hvis vi har landmark i frame.
    Dvs at vi kun gætter på robottens position når vi faktisk har et landmark i framen. 
     """
    velocity, angular_velocity = u_t
    objectIDs, dists, angles = z_t
    particles = state["particles"]

    if velocity != 0.0 or angular_velocity != 0.0:
        sample_motion_model_velocity(particles, velocity, angular_velocity, delta_t)
    print("MCL DEBUG:")
    print("objectIDs:", objectIDs)
    print("in_pause:", state["in_pause"])
    print("measured_this_pause:", state["measured_this_pause"])
    if objectIDs and state["in_pause"] and not state["measured_this_pause"]:
        update_particle_weights(particles, objectIDs, dists, angles, landmarks)
        print("========== MCL WEIGHT UPDATE ==========")
        print("objectIDs:", objectIDs)
        print("dists:", dists)
        print("angles:", angles)
        print("number of particles:", len(particles))

        update_particle_weights(particles, objectIDs, dists, angles, landmarks)

        weights = np.array([p.getWeight() for p in particles])

        print("========== AFTER WEIGHT UPDATE ==========")
        print("max weight:", np.max(weights))
        print("mean weight:", np.mean(weights))
        print("nonzero:", np.count_nonzero(weights))

        best = np.argmax(weights)

        print(
            "BEST PARTICLE:",
            "x =", particles[best].getX(),
            "y =", particles[best].getY(),
            "theta =", np.rad2deg(particles[best].getTheta()),
            "weight =", particles[best].getWeight()
        )

        print("========== NOW RESAMPLING ==========")
        particles = resample_particles(particles)
        xs = np.array([p.getX() for p in particles])
        ys = np.array([p.getY() for p in particles])
        thetas = np.array([p.getTheta() for p in particles])

        print("========== AFTER RESAMPLING CLOUD ==========")
        print("x mean:", np.mean(xs))
        print("x median:", np.median(xs))
        print("x std:", np.std(xs))

        print("y mean:", np.mean(ys))
        print("y median:", np.median(ys))
        print("y std:", np.std(ys))

        print(
            "theta mean:",
            np.rad2deg(
                np.arctan2(
                    np.mean(np.sin(thetas)),
                    np.mean(np.cos(thetas))
                )
            )
        )
        
        state["measured_this_pause"] = True


    return particles

def object_filtering(objects_from_frame, VALID_IDS): #Valid IDs er de ID vi ønsker at selflocalize ud fra
    
    detected_objectIDs, detected_dists, detected_angles=objects_from_frame
    
    # if not isinstance(detected_objectIDs, type(None)):
    #         # List detected objects
    #         for i in range(len(detected_objectIDs)):
    #             print("Object ID = ", detected_objectIDs[i], ", Distance = ", detected_dists[i], ", angle = ", detected_angles[i])

    objectIDs = []
    dists = []
    angles = []

    #Observationer 
    if detected_objectIDs is not None:

        observations = {}

        for ID, dist, angle in zip(
            detected_objectIDs,
            detected_dists,
            detected_angles
        ):
            if ID not in VALID_IDS:
                continue

            # Hvis samme ID ses flere gange,
            # behold den korteste distance
            if ID not in observations or dist < observations[ID][0]:
                observations[ID] = (dist, angle)

        for ID, (measured_dist, measured_angle) in observations.items():

            objectIDs.append(ID)
            dists.append(measured_dist)
            angles.append(measured_angle)
            

    return objectIDs, dists, angles
            
#Funktion til at initialisere inden vi har kørt, her resetter man altså partiklerne og estimatet
def initialize_mcl(num_particles):
    particles = initialize_particles(num_particles)

    est_pose = particle.estimate_pose(particles)

    state = {
        "in_pause": False,
        "measured_this_pause": False,
        "particles": particles,
        "est_pose": est_pose,

        "last_time": time.time(),

        "last_mcl_velocity": 0.0,
        "last_mcl_angular_velocity": 0.0,
        
        "scan_timer": 0.0,
        "scan_rotate_time": 0.0,

        #Kan udkommenteres hvis vi ikke vil gemme billed
        "world": np.zeros((500, 500, 3), dtype=np.uint8),
    }
    

    return state


def scan_function(state,robot_controller, delta_t):
    # Start en ny rotation
    if state["scan_timer"] == 0.0:
        scan_angle = random.uniform(5, 60)
        state["scan_rotate_time"] = np.deg2rad(scan_angle) / abs(W_CALIB)
        state["in_pause"] = False
        state["measured_this_pause"] = False
        print("Ny scan:", scan_angle, "grader")

    state["scan_timer"] += delta_t


    # ROTATION
    if state["scan_timer"] < state["scan_rotate_time"]:
        state["in_pause"] = False
        robot_controller.moters.move(-1, 1)
        state["last_mcl_velocity"] = 0.0
        state["last_mcl_angular_velocity"] = W_CALIB

    # PAUSE
    elif state["scan_timer"] < state["scan_rotate_time"] + SCAN_PAUSE_TIME:
        state["in_pause"] = True
        robot_controller.moters.stop()
        state["last_mcl_velocity"] = 0.0
        state["last_mcl_angular_velocity"] = 0.0

    # NY SCAN
    else:
        state["in_pause"] = False
        robot_controller.moters.stop()
        state["last_mcl_velocity"] = 0.0
        state["last_mcl_angular_velocity"] = 0.0
        state["scan_timer"] = 0.0
        
    # # Start en ny rotation
    # if state["scan_timer"] == 0.0:
    #     state["measured_this_pause"] = False

    #     scan_angle = random.uniform(5, 60)

    #     state["scan_rotate_time"] = (np.deg2rad(scan_angle) / abs(W_CALIB))

    #     print("Ny scan:", scan_angle, "grader")

    # # Opdater timer
    # state["scan_timer"] += delta_t

    # # ROTATION
    # if state["scan_timer"] < state["scan_rotate_time"]:

    #     robot_controller.go_diff_calibrated(-1, 1)

    #     state["last_mcl_velocity"] = 0.0
    #     state["last_mcl_angular_velocity"] = W_CALIB

    # # PAUSE
    # elif state["scan_timer"] < (
    #     state["scan_rotate_time"] + SCAN_PAUSE_TIME
    # ):

    #     robot_controller.stop()

    #     state["last_mcl_velocity"] = 0.0
    #     state["last_mcl_angular_velocity"] = 0.0

    # # NY SCAN
    # else:

    #     robot_controller.stop()

    #     state["last_mcl_velocity"] = 0.0
    #     state["last_mcl_angular_velocity"] = 0.0

    #     state["scan_timer"] = 0.0
            



#Udfører én mcl update ud fra de objecter der detectes i frame og de objecter der ønskes localisering ud fra.
def mcl_update(state:dict, detected_obj:tuple, VALID_IDS: list , landmarks: dict):
    current_time = time.time()

    delta_t = current_time - state["last_time"]
    state["last_time"] = current_time

    # Beskyt mod mærkelige tidsværdier
    delta_t = max(0.0, min(delta_t, 0.5))
    
    #MCL step
    u_t = (state["last_mcl_velocity"],state["last_mcl_angular_velocity"])
    z_t = object_filtering(detected_obj,VALID_IDS)
    
    
    print("OBJECTs:", z_t)
    state["particles"] = mcl_step(state, u_t, z_t, delta_t, landmarks)
    #state["particles"] = mcl_step(state["particles"],u_t,z_t,delta_t,landmarks)
    
    state["est_pose"] = particle.estimate_pose(state["particles"])
    print("Estimate Robot Pose:", state["est_pose"].getX(), state["est_pose"].getY(),state["est_pose"].getTheta() )
    
    #Tegner og gemmer billed af world
    draw_world(state["est_pose"], state["particles"], state["world"])
    cv2.imwrite("mcl_world.png", state["world"])

    return  delta_t



current_state=initialize_mcl(num_particles=1500)

# # TEST AF PARTICLE WEIGHTS
# test_particles = [
#     particle.Particle(x=55, y=50, theta=np.deg2rad(90)),
#     particle.Particle(x=205, y=48, theta=np.deg2rad(44)),
# ]

# update_particle_weights(
#     test_particles,
#     [6, 7],
#     [217.6, 219.1],
#     [np.deg2rad(14.3), np.deg2rad(-21.8)],
#     landmarks
# )

# for p in test_particles:
#     print(
#         "TEST PARTICLE:",
#         "x =", p.getX(),
#         "y =", p.getY(),
#         "theta =", np.rad2deg(p.getTheta()),
#         "weight =", p.getWeight()
#     )


for _ in range(50):
    
    landmarks_seen = robot_controller.camera.detect_landmarks()

    detected_objectIDs = [lm.id for lm in landmarks_seen]
    detected_dists = [lm.distance * 100.0 for lm in landmarks_seen]
    detected_angles    = [np.deg2rad(lm.angle) for lm in landmarks_seen]

    detected_obj =(detected_objectIDs, detected_dists, detected_angles)

    delta_t = mcl_update(
        current_state,
        detected_obj,
        landmarkIDs,landmarks
    )

    # 3. Scan bestemmer bevægelsen for NÆSTE loop
    scan_function(
        current_state,
        robot_controller,
        delta_t
    )


# robot_controller.moters.drive(1)

#current_state["last_mcl_velocity"] = V_CALIB
        
# for _ in range(50):

# landmarks_seen = robot_controller.camera.detect_landmarks()

# detected_objectIDs = [lm.id for lm in landmarks_seen]
# detected_dists = [lm.distance * 100.0 for lm in landmarks_seen]
# detected_angles    = [np.deg2rad(lm.angle) for lm in landmarks_seen]

# detected_obj =(detected_objectIDs, detected_dists, detected_angles)

# # #     delta_t = mcl_update(
# # #         current_state,
# # #         detected_obj,
# # #         landmarkIDs,landmarks
# # #     )

# # #     # 3. Scan bestemmer bevægelsen for NÆSTE loop
# # #     scan_function(
# # #         current_state,
# # #         robot_controller,
# # #         delta_t
# # #     )

# # robot_x = 55.0
# # robot_y = 50.0

# # for ID in landmarkIDs:
# #     lx, ly = landmarks[ID]

# #     expected_dist = np.sqrt(
# #         (lx - robot_x)**2 +
# #         (ly - robot_y)**2
# #     )

# #     # Find det tilsvarende landmark fra kameraet
# #     landmark = next(
# #         (lm for lm in landmarks_seen if lm.id == ID),
# #         None
# #     )

# #     if landmark is not None:
# #         center = landmark.center

# #         print(
# #             f"Landmark: {ID}"
# #             f" | Expected distance: {expected_dist:.2f} cm"
# #             f" | Measured distance: {landmark.distance * 100:.2f} cm"
# #             f" | Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})"
# #         )
        
# while True:
#     delta_t = mcl_update(current_state, detected_obj, landmarkIDs,landmarks)

# # landmarks_seen = robot_controller.camera.detect_landmarks()

# # for lm in landmarks_seen:
# #     print(
# #         f"ID {lm.id}: "
# #         f"center=({lm.center[0]:.3f}, {lm.center[1]:.3f}, {lm.center[2]:.3f}) "
# #         f"distance={lm.distance*100:.2f} cm "
# #         f"angle={lm.angle:.2f}°"
# #     )

