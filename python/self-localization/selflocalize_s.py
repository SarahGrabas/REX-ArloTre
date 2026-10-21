import cv2
import particle
import camera
import numpy as np
from timeit import default_timer as timer
import sys
from motion_model import sample_motion_model_velocity
from math import pi as PI
import time


# Flags
showGUI = False  # Whether or not to open GUI windows
onRobot = True  # Whether or not we are running on the Arlo robot


def isRunningOnArlo():
    """Return True if we are running on Arlo, otherwise False.
      You can use this flag to switch the code from running on you laptop to Arlo - you need to do the programming here!
    """
    return onRobot


robot_module = None
if isRunningOnArlo():
    # XXX: You need to change this path to point to where your robot.py file is located
    sys.path.append("../python")
    try:
        import robot as robot_module
    except ImportError:
        print("selflocalize.py: robot module not present - forcing not running on Arlo!")
        onRobot = False




# Some color constants in BGR format
CRED = (0, 0, 255)
CGREEN = (0, 255, 0)
CBLUE = (255, 0, 0)
CCYAN = (255, 255, 0)
CYELLOW = (0, 255, 255)
CMAGENTA = (255, 0, 255)
CWHITE = (255, 255, 255)
CBLACK = (0, 0, 0)

# Landmarks.
# The robot knows the position of 2 landmarks. Their coordinates are in the unit centimeters [cm].
landmarkIDs = [2]
landmarks = {
    2: (0.0, 0.0),  # Coordinates for landmark 1
    #4: (100.0, 0.0)  # Coordinates for landmark 2
}
landmark_colors = [CRED, CGREEN] # Colors used when drawing the landmarks
target_x, target_y = 50.0, 0.0 #MÅL: midten mellem landmarks

WHEEL_2_WHEEL_DIST = 0.381

meters_pr_sec = 2 * PI * WHEEL_2_WHEEL_DIST / 8 * (3.14/3)
degrees_pr_second = 360 / 4

#V_CALIB = 40.0   # Den kører 40 cm pr. sekund ca. Fremadrettet hastighed ved go_diff_calibrated(1, 1)
#W_CALIB = 1.9234  # I radianer/pr. sekund. Rotationshastighed (~110,2 degree/s) ved go_diff_calibrated(-1, 1)

V_CALIB = meters_pr_sec * 100
W_CALIB = np.deg2rad(degrees_pr_second)

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

# Simon, funktioner til vægte og resampling af partikler
def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


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

        #Landmarkets position i world 
            x_obj, y_obj = landmarks[objectIDs[i]]

            diff_x = x_obj - x
            diff_y = y_obj - y

            particle_dist = np.sqrt(diff_x**2 + diff_y**2)

        #Kameraets målte afstand til landmark
            measured_dist = dists[i]

        #Gaussian probability
        #particle_dist - measured_dist er forskellen mellem opringelig afstand og målte afstand hvis robotten er på denne partikle.
            land_dist = (
                1 / (np.sqrt(2 * np.pi) * sigma_dist)) * np.exp(-0.5 * ((particle_dist - measured_dist) / sigma_dist)**2)
            
            
        #Nu gør vi med vinklerne 
            enheds_theta=np.array([np.cos(theta), np.sin(theta)])
            enheds_particle=np.array([diff_x, diff_y])/np.linalg.norm(np.array([diff_x, diff_y]))
            
            dot_product = np.dot(enheds_theta, enheds_particle)
            dot_product = np.clip(dot_product, -1.0, 1.0)

            theta_particle = np.arccos(dot_product)     
            
            #Vi skal finde ud af om theta_p er negativ eller positiv, dvs om den ligger på højre eller venstre side af enheds theta
            #enheds_theta_hat=np.array(np.sin(theta),-np.cos(theta))
            enheds_theta_hat=np.array([-np.sin(theta),np.cos(theta)])
            
            
            
            final_theta_particle=np.sign(np.dot(enheds_particle, enheds_theta_hat))*theta_particle
            
      
            land_angle = (
                1 / (np.sqrt(2 * np.pi) * sigma_angle)) * np.exp(-0.5 * ((final_theta_particle - angles[i]) / sigma_angle)**2)
            
            land_weight=land_dist*land_angle
        

        #Hvis der er flere landmarks, ganges deres sandsynligheder sammen
            weight *= land_weight
        
        particle_weights.append(weight)    
        
    particle_weights = np.array(particle_weights,dtype=float)

    total_weight = np.sum(particle_weights)

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

# def resample_particles(particles):
#     """We use current weight for particles given in the update particle function.
#     We randomly select same number of particles as the set we sample from.
#     We update the particle list by replacing previous particles with new particles, and we set the weight for each particle to be equal"""
#     weights = np.asarray([p.getWeight() for p in particles], dtype=float)
#     selected = np.random.choice(len(particles), size=len(particles), replace=True, p=weights)
#     return [
#         particle.Particle(
#             particles[index].getX(),
#             particles[index].getY(),
#             particles[index].getTheta(),
#             particles[index].setWeight(1.0 / len(particles)),
#         )
#         for index in selected
#     ]
    
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

TARGET_TOLERANCE = 15.0   # cm
ANGLE_TOLERANCE = 0.15    # rad ≈ 8.6°

def autonomous_controller(
    drive_state,
    estimated_pose,
    target,
    seen_landmarks
):
    x = estimated_pose.getX()
    y = estimated_pose.getY()
    theta = estimated_pose.getTheta()

    target_x, target_y = target

    dx = target_x - x
    dy = target_y - y

    distance = np.hypot(dx, dy)

    target_angle = np.arctan2(dy, dx)

    angle_error = np.arctan2(
        np.sin(target_angle - theta),
        np.cos(target_angle - theta)
    )

    #Hvis vi er indenfor mål tolerancen, dvs. meget tæt på målet stopper vi
    if distance < TARGET_TOLERANCE:
        return "STOP", "STOP"

    #Vi scanner indtil begge landmarks er blevet set
    if drive_state == "SCAN":

        # Vi scanner indtil begge landmarks er fundet
        if len(seen_landmarks) >= 2:
            return "ROTATE_TO_TARGET", "ROTATE_TO_TARGET"

        return "SCAN", "SCAN"


    if drive_state == "ROTATE_TO_TARGET":

        if abs(angle_error) < ANGLE_TOLERANCE:
            return "DRIVE_TO_TARGET", "DRIVE_TO_TARGET"

        if angle_error > 0:
            return "ROTATE_LEFT", "ROTATE_TO_TARGET"

        return "ROTATE_RIGHT", "ROTATE_TO_TARGET"


    if drive_state == "DRIVE_TO_TARGET":

        # Hvis vi begynder at pege for meget forkert,
        # så stop fremdriften og korrigér.
        if abs(angle_error) > ANGLE_TOLERANCE:

            if angle_error > 0:
                return "ROTATE_LEFT", "ROTATE_TO_TARGET"

            return "ROTATE_RIGHT", "ROTATE_TO_TARGET"

        return "DRIVE_FORWARD", "DRIVE_TO_TARGET"


    return "STOP", "STOP"

def execute_robot_command(robot, command):
    """
    Udfører én ikke-blokerende robotkommando.

    Returnerer den faktiske bevægelse:
        velocity       [cm/s]
        angular_velocity [rad/s]

    som robotten forventes at have under kommandoen.
    """

    if command == "STOP":

        robot.stop()

        return 0.0, 0.0

    elif command == "DRIVE_FORWARD":

        robot.go_diff_calibrated(1, 1)

        return V_CALIB, 0.0

    elif command == "ROTATE_LEFT":

        robot.go_diff_calibrated(-1, 1)

        return 0.0, W_CALIB

    elif command == "ROTATE_RIGHT":

        robot.go_diff_calibrated(1, -1)

        return 0.0, -W_CALIB

    return 0.0, 0.0


# Main program #
if __name__ == "__main__":
    cam = None
    try:
        if showGUI:
            # Open windows
            WIN_RF1 = "Robot view"
            cv2.namedWindow(WIN_RF1)
            cv2.moveWindow(WIN_RF1, 50, 50)

            WIN_World = "World view"
            cv2.namedWindow(WIN_World)
            cv2.moveWindow(WIN_World, 500, 50)


        # Initialize particles
        num_particles = 1500
        
        particles = initialize_particles(num_particles)
        print("INITIAL WEIGHT:", particles[0].getWeight())
        print("INITIAL X:", particles[0].getX())
        print("INITIAL Y:", particles[0].getY())
        print("INITIAL THETA:", particles[0].getTheta())

        est_pose = particle.estimate_pose(particles) # The estimate of the robots current pose

        # Driving parameters
        velocity = 0.0
        angular_velocity = 0.0
        drive_state = "SCAN"

        # Den kommando robotten faktisk kørte med
        last_mcl_velocity = 0.0
        last_mcl_angular_velocity = 0.0

        last_time = time.time()

        # Scan state
        scan_direction = 1
        scan_timer = 0.0

        # Initialize the robot (XXX: You do this)
        robot_controller = robot_module.Robot() if isRunningOnArlo() else None

        # Allocate space for world map
        world = np.zeros((500,500,3), dtype=np.uint8)

        # Draw map
        #draw_world(est_pose, particles, world)

        print("Opening and initializing camera")
        if isRunningOnArlo():
            #cam = camera.Camera(0, robottype='arlo', useCaptureThread=True)
            cam = camera.Camera(0, robottype='arlo', useCaptureThread=False)
        else:
            cam = camera.Camera(0, robottype='macbookpro', useCaptureThread=True)
            #cam = camera.Camera(1, robottype='macbookpro', useCaptureThread=False)


        # velocity = 0.0
        # angular_velocity = 0.0
        # drive_state = "SCAN"
        # last_time = timer()
        
        scan_direction = 1 #Om vi drejer til højre eller venstre når vi scanner.
        scan_timer = 0.0

        import random

        SCAN_ROTATE_TIME = random.uniform(0.0, 2.3) #Når vi er i scan mode vælger vi tilfældig vinkel at dreje 
        SCAN_PAUSE_TIME = 0.50
        
        seen_landmarks = set()#Vi gemmer vores observationer her, men altid den tætteste dublet den ser.

        
        while True:

            #action = cv2.waitKey(10)
            #if action == ord('q'): # Quit
                #break

            current_time = time.time()
            delta_t = current_time - last_time #hvor lang tid der er gået siden sidste MCL opdatering og ikke køretiden
            last_time = current_time

            # Beskyt mod mærkelige tidsværdier
            delta_t = max(0.0, min(delta_t, 0.5))

            objectIDs = []
            dists = []
            angles = []
            # Hent kamerabillede og detekter ArUco-mærker (giver z_t)
            colour = cam.get_next_frame()
            detected_objectIDs, detected_dists, detected_angles = cam.detect_aruco_objects(colour)
            if not isinstance(detected_objectIDs, type(None)):
                # List detected objects
                for i in range(len(detected_objectIDs)):
                    print("Object ID = ", detected_objectIDs[i], ", Distance = ", detected_dists[i], ", angle = ", detected_angles[i])
            
            # Nulstil MCL-observationer for hvert loop
        

            if detected_objectIDs is not None:

                VALID_IDS = {2, 4}
                observations = {}

                for ID, dist, angle in zip(
                    detected_objectIDs,
                    detected_dists,
                    detected_angles
                ):

                    if ID not in VALID_IDS:
                        continue

                    # Hvis samme ID ses flere gange,
                    # vælg den korteste distance
                    if ID not in observations or dist < observations[ID][0]:
                        observations[ID] = (dist, angle)

                print("observations:", observations)

                # Lav MCL-arrays ud fra observations
                for ID, (measured_dist, measured_angle) in observations.items():

                    objectIDs.append(ID)
                    dists.append(measured_dist)
                    angles.append(measured_angle)

                    seen_landmarks.add(ID)


            # MCL
            u_t = (
                last_mcl_velocity,
                last_mcl_angular_velocity
            )

            z_t = (objectIDs, dists, angles)

            particles = mcl_step( particles,u_t,z_t,delta_t)
            # objectIDs=[]    
            # dists=[]
            # angles=[]    
                        
            # if detected_objectIDs is not None:   
            #     VALID_IDS = {2, 4}
            #     observations={}
            #         #hvis vi har dubletter af samme id, vælger vi tætteste distance
            #     for ID, dist, angle in zip(detected_objectIDs, detected_dists, detected_angles):
                        
            #         if ID not in VALID_IDS:
            #                 continue
                        
            #         if ID not in observations or dist < observations[ID][0]:
            #                 observations[ID] = (dist, angle)
            #                 print("observations:", observations)

                
            #     for ID, (measured_dist, measured_angle) in observations.items():
            #             objectIDs.append(ID)
            #             dists.append(measured_dist)
            #             angles.append(measured_angle)
                        
            #             seen_landmarks.add(ID)
                    

            # # MCL-KALD
            # u_t = (last_mcl_velocity, last_mcl_angular_velocity) #vi bruger last velocity så det kun er bevægelse i det loop der svarer til delta t tiden.
            #                                                     #så når der ikke er nogle bvægelse kaldes motion model ikke.
            # z_t = (objectIDs, dists, angles)
            # particles = mcl_step(particles, u_t, z_t, delta_t)
            
            

            # Tegn detekterede ArUco-mærker på kamerabilledet hvis fundet
            if objectIDs:
                cam.draw_aruco_objects(colour)

            # Beregn robottens estimerede position efter MCL-opdateringen
            est_pose = particle.estimate_pose(particles)
            none_weights = 0

            for i, p in enumerate(particles):
                if p.getWeight() is None:
                    none_weights += 1

            print("Particles with None weight:", none_weights)
            
            draw_world(est_pose, particles, world)
            cv2.imwrite("mcl_world.png", world)
        
            command, drive_state = autonomous_controller(
                    drive_state,
                    est_pose,
                    (target_x, target_y),
                    seen_landmarks
                )
            
            print("seen objects", seen_landmarks)
            
            print("drive state", drive_state)
            print("objects", detected_objectIDs, detected_dists, detected_angles)
            print("velocity", last_mcl_velocity)
            print("ang velocity", last_mcl_angular_velocity)
            print("robot pose", round(est_pose.getX(),3), round(est_pose.getY(),2),round(est_pose.getTheta(),2))
            
            if isRunningOnArlo():

                if command == "SCAN":

                    # Ny rotation starter
                    if scan_timer == 0.0:

                        scan_angle = random.uniform(5, 60)

                        # Tid der kræves for den valgte vinkel
                        scan_rotate_time = np.deg2rad(scan_angle) / abs(W_CALIB)

                        print("SCAN angle:", round(scan_angle, 2))
                        print("SCAN rotate time:", round(scan_rotate_time, 2))
                        print("SCAN direction:", scan_direction)

                    scan_timer += delta_t

                    #Rotate
                    if scan_timer < scan_rotate_time:

                        if scan_direction > 0:

                            robot_controller.go_diff_calibrated(-1, 1)

                            last_mcl_velocity = 0.0
                            last_mcl_angular_velocity = W_CALIB

                        else:

                            robot_controller.go_diff_calibrated(1, -1)

                            last_mcl_velocity = 0.0
                            last_mcl_angular_velocity = -W_CALIB

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

                # else:
                #     start=timer()
                #     while timer() - start<1:
                #         robot_controller.go_diff_calibrated(1, 1) 
                    
                #     robot_controller.stop()
                    
                    
                #     cv2.imwrite("mcl_world_going.png", world)

            #if showGUI:
                #draw_world(est_pose, particles, world)
                #cv2.imshow(WIN_RF1, colour)
                #cv2.imshow(WIN_World, world)
            
            #Vi stopper nå estimeret position for robotten er ved mål 
            if np.hypot(est_pose.getX() - target_x,est_pose.getY() - target_y) < 10.0:
                break
                
    
    finally: 
        # Make sure to clean up even if an exception occurred
        
        # Close all windows
        cv2.destroyAllWindows()

        # Clean-up capture thread
        if cam is not None:
            cam.terminateCaptureThread()

