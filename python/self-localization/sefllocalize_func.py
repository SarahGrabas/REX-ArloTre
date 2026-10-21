import cv2
import particle
import camera
import numpy as np
from timeit import default_timer as timer
import sys
from math import pi as PI
import time
import robot
import robot as robot_module

import random
#Comment
# Some color constants in BGR format
CRED = (0, 0, 255)
CGREEN = (0, 255, 0)
CBLUE = (255, 0, 0)
CCYAN = (255, 255, 0)
CYELLOW = (0, 255, 255)
CMAGENTA = (255, 0, 255)
CWHITE = (255, 255, 255)
CBLACK = (0, 0, 0)

landmark_colors = [CRED, CGREEN] # Colors used when drawing the landmarks
def jet(x):
    """Colour map for drawing particles. This function determines the colour of 
    a particle from its weight."""
    r = (x >= 3.0/8.0 and x < 5.0/8.0) * (4.0 * x - 3.0/2.0) + (x >= 5.0/8.0 and x < 7.0/8.0) + (x >= 7.0/8.0) * (-4.0 * x + 9.0/2.0)
    g = (x >= 1.0/8.0 and x < 3.0/8.0) * (4.0 * x - 1.0/2.0) + (x >= 3.0/8.0 and x < 5.0/8.0) + (x >= 5.0/8.0 and x < 7.0/8.0) * (-4.0 * x + 7.0/2.0)
    b = (x < 1.0/8.0) * (4.0 * x + 1.0/2.0) + (x >= 1.0/8.0 and x < 3.0/8.0) + (x >= 3.0/8.0 and x < 5.0/8.0) * (-4.0 * x + 5.0/2.0)

    return (255.0*r, 255.0*g, 255.0*b)

def draw_world(est_pose, particles, world):
    """Visualization.
    This functions draws robots position in the world coordinate system."""

    # Fix the origin of the coordinate system
    offsetX = 100
    offsetY = 250

    # Constant needed for transforming from world coordinates to screen coordinates (flip the y-axis)
    ymax = world.shape[0]

    world[:] = CWHITE # Clear background to white

    # Find largest weight
    max_weight = 0
    for particle in particles:
            
        max_weight = max(max_weight, particle.getWeight())

    # Draw particles
    for particle in particles:
        x = int(particle.getX() + offsetX)
        y = ymax - (int(particle.getY() + offsetY))
        colour = jet(particle.getWeight() / max_weight)
        cv2.circle(world, (x,y), 2, colour, 2)
        b = (int(particle.getX() + 15.0*np.cos(particle.getTheta()))+offsetX, 
                                     ymax - (int(particle.getY() + 15.0*np.sin(particle.getTheta()))+offsetY))
        cv2.line(world, (x,y), b, colour, 2)

    # Draw landmarks
    for i in range(len(landmarkIDs)):
        ID = landmarkIDs[i]
        lm = (int(landmarks[ID][0] + offsetX), int(ymax - (landmarks[ID][1] + offsetY)))
        cv2.circle(world, lm, 5, landmark_colors[i], 2)

    # Draw estimated robot pose
    a = (int(est_pose.getX())+offsetX, ymax-(int(est_pose.getY())+offsetY))
    b = (int(est_pose.getX() + 15.0*np.cos(est_pose.getTheta()))+offsetX, 
         ymax-(int(est_pose.getY() + 15.0*np.sin(est_pose.getTheta()))+offsetY))
    cv2.circle(world, a, 5, CMAGENTA, 2)
    cv2.line(world, a, b, CMAGENTA, 2)



def initialize_particles(num_particles):
    particles = []
    for i in range(num_particles):
        # Random starting points. 
        p = particle.Particle(600.0*np.random.ranf() - 100.0, 600.0*np.random.ranf() - 250.0, np.mod(2.0*np.pi*np.random.ranf(), 2.0*np.pi), 1.0/num_particles)
        particles.append(p)

    return particles

meters_pr_sec = 2 * PI * robot.WHEEL_2_WHEEL_DIST / 8 * (3.14/3)
degrees_pr_second = 360 / 4

#V_CALIB = 40.0   # Den kører 40 cm pr. sekund ca. Fremadrettet hastighed ved go_diff_calibrated(1, 1)
#W_CALIB = 1.9234  # I radianer/pr. sekund. Rotationshastighed (~110,2 degree/s) ved go_diff_calibrated(-1, 1)

V_CALIB = meters_pr_sec * 100
W_CALIB = np.deg2rad(degrees_pr_second)

SCAN_PAUSE_TIME = 0.30

cam = None
print("Opening and initializing camera")
cam = camera.Camera(0, robottype='arlo', useCaptureThread=False)

robot_controller = robot_module.Robot()

# Landmarks.
# The robot knows the position of the 4 landmarks. Their coordinates are in the unit centimeters [cm].
landmarkIDs = [2,4,7]
#Coordinates for each landmark
landmarks = {
    2: (0.0, 0.0),
    4: (0.0,100.0),
    7: (100.0,250.0),
    #6: (0.0,250.0)
    
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
    a6 = 0.20

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

    new_particles = []

    # De 90 % resamples fra de eksisterende partikler
    for index in selected:
        new_particle = particle.Particle(
            particles[index].getX(),
            particles[index].getY(),
            particles[index].getTheta(),
            1.0 / num_particles
        )
        new_particles.append(new_particle)

    # De sidste 10 % bliver helt nye random partikler
    random_particles = initialize_particles(num_random)
    new_particles.extend(random_particles)

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
    
    
def mcl_step(particles, u_t, z_t, delta_t,landmarks):
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
            angles,landmarks
        )
    
        particles = resample_particles(particles)


    return particles

def object_filtering(objects_from_frame, VALID_IDS): #Valid IDs er de ID vi ønsker at selflocalize ud fra
    
    detected_objectIDs, detected_dists, detected_angles=objects_from_frame
    
    if not isinstance(detected_objectIDs, type(None)):
            # List detected objects
            for i in range(len(detected_objectIDs)):
                print("Object ID = ", detected_objectIDs[i], ", Distance = ", detected_dists[i], ", angle = ", detected_angles[i])

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
def initialize_mcl(num_particles=1500):
    particles = initialize_particles(num_particles)

    est_pose = particle.estimate_pose(particles)

    state = {
        
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

        state["scan_rotate_time"] = (np.deg2rad(scan_angle) / abs(W_CALIB))

        print("Ny scan:", scan_angle, "grader")

    # Opdater timer
    state["scan_timer"] += delta_t

    # ROTATION
    if state["scan_timer"] < state["scan_rotate_time"]:

        robot_controller.go_diff_calibrated(-1, 1)

        state["last_mcl_velocity"] = 0.0
        state["last_mcl_angular_velocity"] = W_CALIB

    # PAUSE
    elif state["scan_timer"] < (
        state["scan_rotate_time"] + SCAN_PAUSE_TIME
    ):

        robot_controller.stop()

        state["last_mcl_velocity"] = 0.0
        state["last_mcl_angular_velocity"] = 0.0

    # NY SCAN
    else:

        robot_controller.stop()

        state["last_mcl_velocity"] = 0.0
        state["last_mcl_angular_velocity"] = 0.0

        state["scan_timer"] = 0.0
            



#Udfører én mcl update ud fra de objecter der detectes i frame og de objecter der ønskes localisering ud fra.
def mcl_update(state:dict, cam, detected_obj:tuple, VALID_IDS: list , landmarks: dict):
    current_time = time.time()

    delta_t = current_time - state["last_time"]
    state["last_time"] = current_time

    # Beskyt mod mærkelige tidsværdier
    delta_t = max(0.0, min(delta_t, 0.5))
    
    #MCL step
    u_t = (state["last_mcl_velocity"],state["last_mcl_angular_velocity"])
    z_t = object_filtering(detected_obj,VALID_IDS)
    
    state["particles"] = mcl_step(state["particles"],u_t,z_t,delta_t,landmarks)
    

    #Estimate for robot position
    state["est_pose"] = particle.estimate_pose(state["particles"])
    print("Estimate Robot Pose:",state["est_pose"].getX(), state["est_pose"].getY() )
    
    #Tegner og gemmer billed af world
    draw_world(state["est_pose"], state["particles"], state["world"])
    cv2.imwrite("mcl_world.png", state["world"])

    return  delta_t



current_state=initialize_mcl(num_particles=1500)

while True:
    
    colour = cam.get_next_frame()
    detected_objectIDs, detected_dists, detected_angles = cam.detect_aruco_objects(colour)
    detected_obj =(detected_objectIDs, detected_dists, detected_angles)

    delta_t = mcl_update(
        current_state,
        cam,
        detected_obj,
        landmarkIDs,landmarks
    )

    # 3. Scan bestemmer bevægelsen for NÆSTE loop
    scan_function(
        current_state,
        robot_controller,
        delta_t
    )