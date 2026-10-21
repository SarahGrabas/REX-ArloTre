import cv2
import particle
import camera
import numpy as np
from timeit import default_timer as timer
import sys
from motion_model import sample_motion_model_velocity
from math import pi as PI
import time
import robot as robot_module
from selflocalize_s import draw_world, jet
import random

WHEEL_2_WHEEL_DIST = 0.381

meters_pr_sec = 2 * PI * WHEEL_2_WHEEL_DIST / 8 * (3.14/3)
degrees_pr_second = 360 / 4

#V_CALIB = 40.0   # Den kører 40 cm pr. sekund ca. Fremadrettet hastighed ved go_diff_calibrated(1, 1)
#W_CALIB = 1.9234  # I radianer/pr. sekund. Rotationshastighed (~110,2 degree/s) ved go_diff_calibrated(-1, 1)

V_CALIB = meters_pr_sec * 100
W_CALIB = np.deg2rad(degrees_pr_second)

SCAN_PAUSE_TIME = 0.50

print("Opening and initializing camera")
cam = camera.Camera(0, robottype='arlo', useCaptureThread=False)

robot_controller = robot_module.Robot()

# Landmarks.
# The robot knows the position of the 4 landmarks. Their coordinates are in the unit centimeters [cm].
landmarkIDs = [1,2,3,4]
#Coordinates for each landmark
landmarks = {
    1: (0.0, 0.0),
    2: (0.0, 0.0),
    3: (0.0, 0.0),  # Coordinates for landmark 1
    4: (100.0, 0.0)  # Coordinates for landmark 2
}

def initialize_particles(num_particles):
    particles = []
    for i in range(num_particles):
        # Random starting points. 
        p = particle.Particle(600.0*np.random.ranf() - 100.0, 600.0*np.random.ranf() - 250.0, np.mod(2.0*np.pi*np.random.ranf(), 2.0*np.pi), 1.0/num_particles)
        particles.append(p)

    return particles


#This is our observation model
def update_particle_weights(particles, objectIDs, dists, angles):
    #Vi beregn weight ved distance og vinkler
    particle_weights =[]
    #Hvis particle filtering ikke konvergerer kan vi ændre på sigma
    sigma_dist = 15.0 #cm
    sigma_angle=0.10
    
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

    weights = np.asarray(
        [p.getWeight() for p in particles],
        dtype=float
    ).flatten()

    # Sørg for at weights summerer til 1
    total = np.sum(weights)

    if total <= 0 or not np.isfinite(total):
        weights = np.ones(len(particles)) / len(particles)
    else:
        weights = weights / total

    print("weights shape:", weights.shape)
    print("weights sum:", np.sum(weights))

    selected = np.random.choice(
        len(particles),
        size=len(particles),
        replace=True,
        p=weights
    )

    new_particles = []

    for index in selected:

        new_particle = particle.Particle(
            particles[index].getX(),
            particles[index].getY(),
            particles[index].getTheta(),
            1.0 / len(particles)
        )

        new_particles.append(new_particle)

    return new_particles

def mcl_step(particles, u_t, z_t, delta_t):
    """
    Funktion der udfører MCL-skridt: Prediction, Correction og Resampling.
    Vi beregner particles movement ud fra velocity og angular verlocity som burde afspejle robottens bevægelse.
    Vi opdaterer partikel vægt og resampler hvis vi har landmark i frame.
    Dvs at vi kun gætter på robottens position når vi faktisk har et landmark i framen. 
     """
    velocity, angular_velocity = u_t
    objectIDs, dists, angles = z_t

    if velocity != 0.0 or angular_velocity != 0.0:
        sample_motion_model_velocity(
            particles,
            velocity,
            angular_velocity,
            delta_t
        )

    if objectIDs:
        update_particle_weights(
            particles,
            objectIDs,
            dists,
            angles
        )
    
        particles = resample_particles(particles)


    return particles

#Funktion til at initialisere inden vi har kørt, her resetter man altså partiklerne og estimatet
def initialize_mcl(num_particles=1500):
    particles = initialize_particles(num_particles)

    est_pose = particle.estimate_pose(particles)

    state = {
        "particles": particles,
        "est_pose": est_pose,

        "last_time": time.time(),

        "last_mcl_velocity": 0.0,
        "last_mcl_angular_velocity": 0.0,

        "drive_state": "SCAN",

        "scan_direction": 1,
        "scan_timer": 0.0,

        "seen_landmarks": set(),
        
        #Kan udkommenteres hvis vi ikke vil gemme billed
        "world": np.zeros((500, 500, 3), dtype=np.uint8),
    }
    

    return state


def scan_function(state,robot_controller, delta_t):
    if state["drive_state"] == "SCAN":

        # Ny rotation starter
        if scan_timer == 0.0:

            scan_angle = random.uniform(5, 60)

            # Tid der kræves for den valgte vinkel
            scan_rotate_time = np.deg2rad(scan_angle) / abs(W_CALIB)

            scan_timer += delta_t

            #Rotate hvis vi er inden for en rotations tid
            if scan_timer < scan_rotate_time:
                
                #Vi drejer kun til venstre     
                robot_controller.go_diff_calibrated(-1, 1)

                last_mcl_velocity = 0.0
                last_mcl_angular_velocity = W_CALIB


            #Pause
            elif scan_timer < scan_rotate_time + SCAN_PAUSE_TIME:

                robot_controller.stop()

                last_mcl_velocity = 0.0
                last_mcl_angular_velocity = 0.0

            #Ny rotation
            else:

                scan_timer = 0.0

                robot_controller.stop()

                last_mcl_velocity = 0.0
                last_mcl_angular_velocity = 0.0
                
            return last_mcl_velocity, last_mcl_angular_velocity


#Udfører én mcl update, ligesom vore while-loop fra før
def mcl_update(state, cam):
    current_time = time.time()

    delta_t = current_time - state["last_time"]
    state["last_time"] = current_time

    # Beskyt mod mærkelige tidsværdier
    delta_t = max(0.0, min(delta_t, 0.5))

   #Detect objects
    colour = cam.get_next_frame()

    detected_objectIDs, detected_dists, detected_angles = cam.detect_aruco_objects(colour)
    if not isinstance(detected_objectIDs, type(None)):
            # List detected objects
            for i in range(len(detected_objectIDs)):
                print("Object ID = ", detected_objectIDs[i], ", Distance = ", detected_dists[i], ", angle = ", detected_angles[i])

    objectIDs = []
    dists = []
    angles = []

    #Observationer 
    if detected_objectIDs is not None:

        VALID_IDS = {i for i in landmarkIDs}
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

            state["seen_landmarks"].add(ID)

    #MCL step
    u_t = (state["last_mcl_velocity"],state["last_mcl_angular_velocity"])
    z_t = (objectIDs, dists,angles)
    state["particles"] = mcl_step(state["particles"],u_t,z_t,delta_t)
    
    # Tegn detekterede ArUco-mærker på kamerabilledet hvis fundet
    if objectIDs:
        cam.draw_aruco_objects(colour)

    #Estimate for robot position
    state["est_pose"] = particle.estimate_pose(
        state["particles"]
    )
    
    #Tegner og gemmer billed af world
    draw_world(state["est_pose"], state["particles"], state["world"])
    cv2.imwrite("mcl_world.png", state["world"])

    return {
        "colour": colour,
        "objectIDs": objectIDs,
        "dists": dists,
        "angles": angles,
        "delta_t": delta_t,
        "est_pose": state["est_pose"],
    }

current_state=initialize_mcl(num_particles=1500)

while True:
    
    current_state = mcl_update(current_state, cam)
    