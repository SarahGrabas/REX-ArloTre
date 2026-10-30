import selfloc_particle_class as particle
import numpy as np
import time

#Motion model, opdatere partikler ud fra motion siden sidste mcl kald + noise
def sample_motion_model(particles, movement):
    """Motion model
       Hvis der har været movement (forward eller bagud) på robotten flyttes/drejes partiklerne med robotten (+ noise).
       Hvis vi har roteret, opdateres partiklernes orientering med vinklen for denne rotation (+ noise).
       Note:
       Det er vigtigt at hvis movement både indeholder driven og degree, at robotten er drejet først og så kørt."""
    driven, degree =movement

    #Noise
    sigma_driven = 8.0   # cm
    sigma_degree = 5.0     #grader

    for p in particles:
        x = p.getX()
        y = p.getY()
        theta = p.getTheta()
        
        if degree !=0:
            degree_hat = degree + np.random.normal(0, sigma_degree) #lægger noise til mellem -5 og 5 grader

            degree_hat = np.deg2rad(degree_hat) #grader til radianer

            theta = np.mod(theta + degree_hat, 2.0 * np.pi) #theta_new , opdaterer orientering
            p.setTheta(theta) #theta_new
        
        if driven !=0:
            driven_hat = driven + np.random.normal(0, sigma_driven) #lægger noise til, svarer til robot har kørt max 8 cm for kort eller for langt
            
            x_new = x + driven_hat * np.cos(theta)  #opdater position i retning af ny orientering
            y_new = y + driven_hat * np.sin(theta)
            p.setX(x_new)
            p.setY(y_new)
        
#Vi generer partikler der ligger inden for banens koordinater
def initialize_particles(num_particles):
    X_MIN = 0.0
    X_MAX = 690.0
    Y_MIN = 0.0
    Y_MAX = 590.0

    particles = []
    for i in range(num_particles):
        x = np.random.uniform(X_MIN, X_MAX) 
        y = np.random.uniform(Y_MIN, Y_MAX)
        theta = np.random.uniform(0, 2*np.pi)

        p = particle.Particle(x, y, theta, 1.0 / num_particles)
        particles.append(p)

    return particles

#Observation model, 
def update_particle_weights(particles, objectIDs, dists, angles,landmarks):
    """VIGTIGT: 'dists' er kameraets målte distance til landmark konverteret til afstanden fra robottens centrum og landmarks centrum.
    Veregner og tildeler en sandsynlighedsvægt til hver partikel ud fra
    robotkameraets målinger af distance og vinkel til landmarks.

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
    #Noise
    sigma_dist = 15.0 #cm
    sigma_angle=0.20    #rad
    
    particle_weights =[]
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
            
            #Vinkel beregninger for oprindelig vinkel fra partikel til landmark
            enheds_theta=np.array([np.cos(theta), np.sin(theta)])
            enheds_particle=np.array([diff_x, diff_y])/np.linalg.norm(np.array([diff_x, diff_y]))
            dot_product = np.dot(enheds_theta, enheds_particle) #prikprodukt
            dot_product = np.clip(dot_product, -1.0, 1.0) #sikrer at vi ikke får tal som 1.00000002
            theta_particle = np.arccos(dot_product)    #oprindelig vinkel uden fortegn
            
            #Vi skal finde ud af om theta_p er negativ eller positiv, dvs om den ligger på højre eller venstre side af enheds theta
            #enheds_theta_hat=np.array(np.sin(theta),-np.cos(theta)) #Denne vektor ville vi skulle bruge hvis kamrea har positiv vinkel til højre
            enheds_theta_hat=np.array([-np.sin(theta),np.cos(theta)])   #Denne vektor bruger vi da kamrea har positiv vinkel til venstre
            final_theta_particle=np.sign(np.dot(enheds_particle, enheds_theta_hat))*theta_particle #oprindelig vinkel med fortegn
            
            #Gaussian probability: (final_theta_particle - angles) er forskellen mellem opringelig vinkel og målte vinkel hvis robotten har denne partikel vinkel.
            land_angle = (1 / (np.sqrt(2 * np.pi) * sigma_angle)) * np.exp(-0.5 * ((final_theta_particle - angles[i]) / sigma_angle)**2)
            
            #For hvert landmark vi observerer ganger vi probability for distance og vinkelmåling
            land_weight=land_dist*land_angle
            
            #Hvis der er flere landmarks, ganges deres sandsynligheder sammen
            weight *= land_weight 
        
        particle_weights.append(weight)    
        
    particle_weights = np.array(particle_weights,dtype=float)
    total_weight = np.sum(particle_weights)

    #Normalisering af vægtene
    if total_weight > 0:
            particle_weights = particle_weights / total_weight
    else:
            particle_weights[:] = 1.0 / len(particle_weights)
        
    for p, p_weight in zip(particles,particle_weights):
            p.setWeight(p_weight)
            
def resample_particles(particles):
    """Funktion til at resample partikler.
        Vi bruger også Random sample injection, hvor vi erstatter 10 % af partiklerne med nye random samples."""
    num_particles = len(particles) 
    num_random = int(0.10 * num_particles) #10 % of particles will be new random samples
    num_resampled = num_particles - num_random #90 % resamples med vægte

    weights = np.asarray([p.getWeight() for p in particles],dtype=float)
    total = np.sum(weights)

    if total <= 0 or not np.isfinite(total):
        weights = np.ones(num_particles) / num_particles
    else:
        weights = weights / total

    #Resampling af 90% af partiklerne med vægte (med replacement)
    selected = np.random.choice(num_particles,size=num_resampled,replace=True,p=weights)

    new_particles = []
    for index in selected:
        new_particle = particle.Particle(particles[index].getX(),particles[index].getY(),particles[index].getTheta(),)
        new_particles.append(new_particle)

    #Sampler 10% som nye random samples
    random_particles = initialize_particles(num_random)
    new_particles.extend(random_particles)

    #Nulstiller vægte
    for p in new_particles:
        p.setWeight(1.0 / num_particles)
        
    particle.add_uncertainty(new_particles, 3.0, 0.05)

    return new_particles

def mcl_step(state: dict, z_t: tuple,landmarks: dict[int, tuple:float]):
    """
    Funktion der udfører MCL-skridt: Prediction, Correction og Resampling.
    Vi opdaterer partikler ud fra robot movement.
    Vi opdaterer partikel vægt og resampler hvis vi har landmark i frame.
     """
    particles = state["particles"]
    driven, degree =state["movement"]

    #Hvis vi enten har kørt eller drejet flytter partikler med
    if driven != 0.0 or degree != 0.0:
        sample_motion_model(particles, state["movement"])
    
    if z_t is not None:
        objectIDs, dists, angles = z_t #Observations we use to update weights
        if objectIDs: 
            update_particle_weights(particles, objectIDs, dists, angles, landmarks)
            particles = resample_particles(particles)

    return particles

def object_filtering(objects_from_frame: tuple, VALID_IDS):
    """Funktion til at at sortere de observationer fra, som vi ikke ønsker at selflocalize ud fra"""
    detected_objectIDs, detected_dists, detected_angles=objects_from_frame

    objectIDs = []
    dists = []
    angles = []
    if detected_objectIDs is not None:
        observations = {}
        for ID, dist, angle in zip(detected_objectIDs, detected_dists,detected_angles):
            if ID not in VALID_IDS:
                continue
            # Hvis samme ID ses flere gange, beholder den korteste distance
            if ID not in observations or dist < observations[ID][0]:
                observations[ID] = (dist, angle)

        for ID, (measured_dist, measured_angle) in observations.items():
            objectIDs.append(ID)
            dists.append(measured_dist)
            angles.append(measured_angle)
            
    return objectIDs, dists, angles
            
#Funktion til at initialisere, her resetter man altså partiklerne og estimatet
def initialize_mcl(num_particles):
    """Funktion til at initalizere partikler og robot position og movement"""
    particles = initialize_particles(num_particles)
    est_pose = particle.estimate_pose(particles)

    state = {
        "particles": particles, 
        "est_pose": est_pose, #Estimate for robot position
        "movement": [0,0], #(cm for- or backward, degree turned (left positive))
    }
    

    return state

def scan_function(state, robot_controller):
    """Funktion der rotere robotten 50 grader mod venstre.
       Bruges under scanning efter landmarks i selflocalization.
       Grader beregnes ud fra FOV, så vi ikke springer noget over når vi roterer.
       Vi opdaterer movement i vores state dictionary, så det passer med faktisk rotation."""
    degree=50 #Vi drejer mindre end FOV, så vi får alt med, FOV er ca 64
    robot_controller.motors.rotate(degree)
    time.sleep(1.0)
    state["movement"][1]+=degree #update movement state

def mcl_update(state:dict, detected_obj:tuple, VALID_IDS: list , landmarks: dict):
    """Funktionen filtrerer objecter, laver mcl og estimerer robot position.
       Robot movement nulstilles også."""
    if detected_obj is not None:
        z_t = object_filtering(detected_obj,VALID_IDS)
        print("OBJECTs for selflocalizing:", z_t)
    else: z_t=None
    
    state["particles"] = mcl_step(state, z_t, landmarks)
    state["est_pose"] = particle.estimate_pose(state["particles"])
    print("Estimate Robot Pose:", state["est_pose"].getX(), state["est_pose"].getY(),state["est_pose"].getTheta() )
    
    #Bevægelsen er nu blevet brugt så vi nulstiller movement
    state["movement"] = [0, 0]

def data_objects(landmarks_seen, valid_lm_seen: list, valid_objects:list): 
    """Funktion til at trække de observations data (id, distance, vinkel) vi skal bruge i selflocalization.
       Denne funktion opdaterer også de landmark corners vi har set indtil videre, så vi kan holde øje med hvornår vi har set 2."""
    detected_objectIDs = [lm.id for lm in landmarks_seen]
    detected_dists = [lm.distance * 100.0 for lm in landmarks_seen]

    detected_angles = [np.deg2rad(lm.angle) for lm in landmarks_seen]

    detected_obj =(detected_objectIDs, detected_dists, detected_angles) #alle objecter i frame
    print(detected_obj)
    
    for i in range(len(detected_objectIDs)):
        if detected_objectIDs[i] in valid_objects and detected_objectIDs[i] not in valid_lm_seen:
            valid_lm_seen.append(detected_objectIDs[i])
    
    return detected_obj, valid_lm_seen

