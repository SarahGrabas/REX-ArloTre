import cv2
import particle
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


    


#Motion model, opdatere partikler ud fra motion kun
def sample_motion_model(particles, movement):
    """Motion model
       Hvis der har været movement (forward eller bagud) flyttes partiklerne med denne movement (+ noise).
       Hvis vi har roteret, opdateres partiklernes orientering med denne rotation (+ noise)"""
    driven, degree =movement

    # Noise
    sigma_driven = 5.0   # cm
    sigma_degree = 2.0     # grader

    for p in particles:

        x = p.getX()
        y = p.getY()
        theta = p.getTheta()
        
        #Hvis vi har kørt, opdatere partikler efter hvor mange cm vi har kørt
        if driven !=0:
            # Tilfældig fejl på movement
            driven_hat = driven + np.random.normal(0, sigma_driven)
            
            # Opdater position
            x_new = x + driven_hat * np.cos(theta)
            y_new = y + driven_hat * np.sin(theta)
            
            p.setX(x_new)
            p.setY(y_new)
        
        if degree !=0:

            # Tilfældig fejl på rotation
            degree_hat = degree + np.random.normal(0, sigma_degree)

            #grader til radianer
            degree_hat = np.deg2rad(degree_hat)

            # Opdater orientering
            theta_new = np.mod(theta + degree_hat, 2.0 * np.pi)

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
    """VIGTIGT: 'dists' er kameraets målte distance til landmark konverteret til afstanden fra robottens centrum og landmarks centrum.
    Beregner og tildeler en sandsynlighedsvægt til hver partikel ud fra
    robotkameraets målinger af landmarks.

    For hver partikel sammenlignes den forventede observation af hvert
    observeret landmark med kameraets faktiske måling. Der beregnes både
    en sandsynlighed for afstanden og en sandsynlighed for vinklen.
    Disse sandsynligheder kombineres til én samlet vægt for partiklen.

    Afstanden beregnes som den euklidiske afstand mellem partiklens
    position (x, y) og landmarkets position (x_obj, y_obj).

    Forskellen mellem den forventede afstand og kameraets målte afstand
    bruges i en Gaussisk sandsynlighedsfordeling.
    
    Vinklen mellem partiklens orientering og kameraets retningen mod landmarket
    beregnes og bruges derefter til at beregne en Gaussisk
    sandsynlighed for kameraets målte vinkel."""
    
    particle_weights =[]
    
    sigma_dist = 15.0 #cm
    sigma_angle=0.20    #rad
    
    for p in particles:
        x = p.getX()
        y = p.getY()
        theta = p.getTheta()
        weight = 1.0 

        for i in range(len(objectIDs)):
            x_obj, y_obj = landmarks[objectIDs[i]]  #Landmarkets coordinater for centrum
            
            diff_x = x_obj - x
            diff_y = y_obj - y
            particle_dist = np.sqrt(diff_x**2 + diff_y**2) #Beregn euclidean distance fra partikel til landmark centrum
            
            measured_dist = dists[i]    #Målte distance fra kamera -> afstand fra robot centrum til landmark centrum

        #Gaussian probability: (particle_dist - measured_dist) er forskellen mellem opringelig afstand og målte afstand hvis robotten er på denne partikle position.
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
            
            #Gaussian probability: (final_theta_particle - angles) er forskellen mellem opringelig vinkel og målte vinkel hvis robotten har denne partikel vinkel.
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
        )
        new_particles.append(new_particle)

    # De sidste 10 % bliver helt nye random partikler
    random_particles = initialize_particles(num_random)
    new_particles.extend(random_particles)

    for p in new_particles:
        p.setWeight(1.0 / num_particles)
        
    particle.add_uncertainty(new_particles, 3.0, 0.05)

    return new_particles

def mcl_step(state: dict,z_t: tuple,landmarks: dict[int, tuple:float]):
    """
    Funktion der udfører MCL-skridt: Prediction, Correction og Resampling.
    Vi beregner particles movement ud fra velocity og angular verlocity som burde afspejle robottens bevægelse.
    Vi opdaterer partikel vægt og resampler hvis vi har landmark i frame.
    Dvs at vi kun gætter på robottens position når vi faktisk har et landmark i framen. 
     """
    objectIDs, dists, angles = z_t #Observations we use to update weights
    particles = state["particles"]
    driven, degree =state["movement"]

    #Hvis vi enten har kørt eller drejet flytte partikler med
    if driven != 0.0 or degree != 0.0:
        sample_motion_model(particles, state["movement"])
 
    if objectIDs: 
        
        update_particle_weights(particles, objectIDs, dists, angles, landmarks)

    
        particles = resample_particles(particles)


    return particles

def object_filtering(objects_from_frame: tuple, VALID_IDS): #Valid IDs er de ID vi ønsker at selflocalize ud fra
    
    detected_objectIDs, detected_dists, detected_angles=objects_from_frame

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
        "particles": particles,
        "est_pose": est_pose, #Estimate for robot position
        "movement": (0,0), #(cm for- or backward, degree turned (left positive))

        #Kan udkommenteres hvis vi ikke vil gemme billed
        "world": np.zeros((500, 500, 3), dtype=np.uint8),
    }
    

    return state


def scan_function(state, robot_controller):
    """Funktion der rotere.
       Bruges under scanning efter landmarks i selflocalization.
       Grader beregnes ud fra FOV, så vi ikke springer noget over når vi roterer. (Laver for store rotationer).
       Vi opdaterer movement i vores state dictionary, så det passer med faktisk rotation."""
    degree=50 #Vi drejer mindre end FOV, så vi får alt med, FOV er ca 64 grader
    robot_controller.moters.rotate(degree)
    time.sleep(1.0)
    state["movement"]=(0,degree) #Update movement state



#Udfører én mcl update ud fra de objecter der detectes i frame og de objecter der ønskes localisering ud fra.
def mcl_update(state:dict, detected_obj:tuple, VALID_IDS: list , landmarks: dict):
    #MCL step
    z_t = object_filtering(detected_obj,VALID_IDS)
    print("OBJECTs:", z_t)
    
    state["particles"] = mcl_step(state, z_t, landmarks)
    #state["particles"] = mcl_step(state["particles"],u_t,z_t,delta_t,landmarks)
    
    state["est_pose"] = particle.estimate_pose(state["particles"])
    print("Estimate Robot Pose:", state["est_pose"].getX(), state["est_pose"].getY(),state["est_pose"].getTheta() )
    
    #Tegner og gemmer billed af world
    # draw_world(state["est_pose"], state["particles"], state["world"])
    # cv2.imwrite("mcl_world.png", state["world"])
    
    # Bevægelsen er nu blevet brugt så vi nulstiller movement
    state["movement"] = (0, 0)



# robot_controller.moters.drive(1) #Vi skal have implementeret at alle motor funktioner opdaterer state["movement"] inden mcl kaldes
# current_state["movement"] = (30,0)

# for _ in range(50):   
#     landmarks_seen = robot_controller.camera.detect_landmarks()

#     detected_objectIDs = [lm.id for lm in landmarks_seen]
#     detected_dists = [lm.distance * 100.0 for lm in landmarks_seen]

#     detected_angles = [np.deg2rad(lm.angle) for lm in landmarks_seen]

#     detected_obj =(detected_objectIDs, detected_dists, detected_angles)
#     print(detected_obj)

#     mcl_update(current_state,detected_obj,landmarkIDs,landmarks)

#     #Her behøver vi ikke nulstille movement, da scan opdaterer den
#     scan_function(current_state,robot_controller)


import Exercise_4_2_rrt as rrt_class   
import Exercise_4_2_grid as grid
import numpy as np
from typing import Sequence, Union
# import matplotlib.pyplot as plt

import json

import robot


#robot_controller = rally_copy.Robot()

arlo_radius=rally.ARLO_RADIUS

def RRT_call(GOAL:tuple[float, float],obstacles: tuple[int,tuple[float,float],float], estimate_robot_pose:tuple[float,float,float]):
    X_MIN= 0.0
    X_MAX=120+22+400+28+120
    Y_MIN= 0.0
    Y_MAX= 120+22+300+28+120
    GRID_CELL_SIZE=0.1
    map = grid.Grid(X_MIN,X_MAX,Y_MIN,Y_MAX, GRID_CELL_SIZE)
    map.obstacles(obstacles, robot_radius=arlo_radius)
    
    START_POINT=[estimate_robot_pose[0],estimate_robot_pose[1]]
    rrt = rrt_class.RRT(start=START_POINT,goal=GOAL,map=map,grid_cell_size=GRID_CELL_SIZE,)
    path = rrt.path_planning()
    
    if path is None:
            print("Cannot find path")
    else:
        print("found path!!")
        print(path)
        simpler_path=rrt.simpler_path(path)
        print(simpler_path)
    
    
    return simpler_path


current_state=initialize_mcl(num_particles=1500)

for i in range(10):   
    landmarks_seen = robot_controller.camera.detect_landmarks()

    detected_objectIDs = [lm.id for lm in landmarks_seen]
    detected_dists = [lm.distance * 100.0 for lm in landmarks_seen]

    detected_angles = [np.deg2rad(lm.angle) for lm in landmarks_seen]
    
    detected_center = [lm.center for lm in landmarks_seen]

    detected_obj =(detected_objectIDs, detected_dists, detected_angles)
    print(detected_obj)
    
    landmarkIDs=[]
    landmarks={}
    for i in range(len(detected_objectIDs)):
        if detected_objectIDs[i] not in landmarkIDs:
            landmarkIDs.append(detected_objectIDs[i])
            landmarks[detected_objectIDs[i]]= detected_center[i]
        
        

    mcl_update(current_state,detected_obj,landmarkIDs,landmarks)

    scan_function(current_state,robot_controller)
    
    if i == 2:
        draw_world(current_state["est_pose"], current_state["particles"], current_state["world"])
        cv2.imwrite("mcl_world_after5.png", current_state["world"])
        
    if i == 5:
        draw_world(current_state["est_pose"], current_state["particles"], current_state["world"])
        cv2.imwrite("mcl_world_after10.png", current_state["world"])
        
    if i == 10:
        draw_world(current_state["est_pose"], current_state["particles"], current_state["world"])
        cv2.imwrite("mcl_world_after20.png", current_state["world"])

obstacles = robot_controller.camera.detect_landmarks()

detected_objectIDs = [lm.id for lm in obstacles]
detected_center = [lm.center  for lm in obstacles]
detected_radius = [lm.radius for lm in obstacles]

detected_obj =[]
for i in range(len(detected_objectIDs)):
    if detected_objectIDs[i] != 1:
        detected_obj.append(detected_objectIDs[i], detected_center[i], detected_radius[i])
    
    if detected_objectIDs[i]== 1:
        GOAL = (detected_center[i])


print(detected_obj)
RRT_call(GOAL, detected_obj,current_state["est_pose"])
    