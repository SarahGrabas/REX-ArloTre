import cv2
import particle
import camera
import numpy as np
from timeit import default_timer as timer
import sys
from motion_model import sample_motion_model_velocity


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
landmarkIDs = [1, 11]
landmarks = {
    1: (0.0, 0.0),  # Coordinates for landmark 1
    11: (100.0, 0.0)  # Coordinates for landmark 2
}
landmark_colors = [CRED, CGREEN] # Colors used when drawing the landmarks
target_x, target_y = 50.0, 0.0 #MÅL: midten mellem landmarks

V_CALIB = 40.0   # Den kører 40 cm pr. sekund ca. Fremadrettet hastighed ved go_diff_calibrated(1, 1)
W_CALIB = 1.9234  # I radianer/pr. sekund. Rotationshastighed (~110,2 degree/s) ved go_diff_calibrated(-1, 1)

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
    weights = np.asarray([p.getWeight() for p in particles], dtype=float)
    selected = np.random.choice(len(particles), size=len(particles), replace=True, p=weights)
    return [
        particle.Particle(
            particles[index].getX(),
            particles[index].getY(),
            particles[index].getTheta(),
            1.0 / len(particles),
        )
        for index in selected
    ]

def mcl_step(particles, u_t, z_t, delta_t):
    """
    Udfører ét komplet MCL-skridt: Prediction, Correction og Resampling.
    """
    velocity, angular_velocity = u_t
    objectIDs, dists, angles = z_t

    # 1. Prediction 
    if velocity != 0.0 or angular_velocity != 0.0:
        sample_motion_model_velocity(particles, velocity, angular_velocity, delta_t)

    # 2. Correction 
    if objectIDs:
        update_particle_weights(
        particles,
        objectIDs,
        dists,
        angles
    )

        particles = resample_particles(particles)

    return particles

# udkast til en tilstandsmaskine ift. at finde midterpunktet mellem to landmarks
def autonomous_controller(est_pose, objectIDs, drive_state, seen_landmarks):
    """
    Beregner motorkommandoer (velocity, angular_velocity) baseret på 
    MCL-estimatet (est_pose) og den aktive tilstand.
    """
    
    # 1. Udregn distancer og vinkel-fejl til målet ud fra MCL-poseringen[cite: 1, 4]
    dx = target_x - est_pose.getX()
    dy = target_y - est_pose.getY()
    dist_to_target = np.hypot(dx, dy)
    
    target_angle = np.arctan2(dy, dx)
    angle_error = wrap_angle(target_angle - est_pose.getTheta())
    
    velocity = 0.0
    angular_velocity = 0.0

    # 2. Tilstandsmaskinens logik. Skal indsætte nogle af de oprindelige funktioner for robotstyring.
    if drive_state == "SCAN":

        velocity = 0.0
        angular_velocity = 0.0

        print("SCAN")

        if 1 in seen_landmarks and 11 in seen_landmarks:
            drive_state = "ROTATE_TO_TARGET"
            
        
        # if not isinstance(objectIDs, type(None)) and len(seen_landmarks) >= 2:
        #     drive_state = "ROTATE_TO_TARGET"

    if drive_state == "ROTATE_TO_TARGET":

        if abs(angle_error) > 0.08:

            if angle_error > 0:
                angular_velocity = W_CALIB
            else:
                angular_velocity = -W_CALIB

            velocity = 0.0

        else:
            angular_velocity = 0.0
            drive_state = "DRIVE_TO_TARGET"

    elif drive_state == "DRIVE_TO_TARGET":

        if dist_to_target > 10.0:

            velocity = V_CALIB

            if abs(angle_error) > 0.08:
                angular_velocity = np.sign(angle_error) * W_CALIB
            else:
                angular_velocity = 0.0

        else:
            velocity = 0.0
            angular_velocity = 0.0
            drive_state = "STOP"

    elif drive_state == "STOP":
        print("STOP")
        velocity = 0.0
        angular_velocity = 0.0

    return velocity, angular_velocity, drive_state

# Main program #
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
    num_particles = 1000
    particles = initialize_particles(num_particles)

    est_pose = particle.estimate_pose(particles) # The estimate of the robots current pose

    # Driving parameters
    velocity = 0.0 # cm/sec
    angular_velocity = 0.0 # radians/sec

    # Initialize the robot (XXX: You do this)
    robot_controller = robot_module.Robot() if isRunningOnArlo() else None

    # Allocate space for world map
    #world = np.zeros((500,500,3), dtype=np.uint8)

    # Draw map
    #draw_world(est_pose, particles, world)

    print("Opening and initializing camera")
    if isRunningOnArlo():
        #cam = camera.Camera(0, robottype='arlo', useCaptureThread=True)
        cam = camera.Camera(0, robottype='arlo', useCaptureThread=False)
    else:
        cam = camera.Camera(0, robottype='macbookpro', useCaptureThread=True)
        #cam = camera.Camera(1, robottype='macbookpro', useCaptureThread=False)

    velocity = 0.0
    angular_velocity = 0.0
    drive_state = "SCAN"
    last_time = timer()
    
    scan_direction = 1
    scan_timer = 0.0

    SCAN_ROTATE_TIME = 1.0
    SCAN_PAUSE_TIME = 0.50
    
    seen_landmarks = set()#Vi gemmer vores observationer her, men altid den tætteste dublet den ser.
    while True:

        objectIDs=[]
        dists=[]
        angles=[]

        action = cv2.waitKey(10)
        if action == ord('q'): # Quit
            break

        current_time = timer()
        delta_t = current_time - last_time
        last_time = current_time

        # Hent kamerabillede og detekter ArUco-mærker (giver z_t)
        colour = cam.get_next_frame()
        detected_objectIDs, detected_dists, detected_angles = cam.detect_aruco_objects(colour)
        if not isinstance(detected_objectIDs, type(None)):
            # List detected objects
            for i in range(len(detected_objectIDs)):
                print("Object ID = ", detected_objectIDs[i], ", Distance = ", detected_dists[i], ", angle = ", detected_angles[i])
            
        if detected_objectIDs is not None:   
            VALID_IDS = {1, 11}
            observations={}
                #hvis vi har dubletter af samme id, vælger vi tætteste distance
            for ID, dist, angle in zip(detected_objectIDs, detected_dists, detected_angles):
                    
                if ID not in VALID_IDS:
                        continue
                    
                if ID not in observations or dist < observations[ID][0]:
                        observations[ID] = (dist, angle)

            
            for ID, (measured_dist, measured_angle) in observations.items():
                    objectIDs.append(ID)
                    dists.append(measured_dist)
                    angles.append(measured_angle)
                    
                    seen_landmarks.add(ID)
                

        # MCL-KALD
        u_t = (velocity, angular_velocity)
        z_t = (objectIDs, dists, angles)
        particles = mcl_step(particles, u_t, z_t, delta_t)

        # Tegn detekterede ArUco-mærker på kamerabilledet hvis fundet
        if not isinstance(objectIDs, type(None)):
            cam.draw_aruco_objects(colour)

        # Beregn robottens estimerede position efter MCL-opdateringen
        est_pose = particle.estimate_pose(particles)

        velocity, angular_velocity, drive_state = autonomous_controller(est_pose, objectIDs, drive_state,seen_landmarks)
        
        
        if isRunningOnArlo():

            if drive_state == "SCAN":

                scan_timer += delta_t

                print(
                    "SCAN timer:",
                    round(scan_timer, 2),
                    "direction:",
                    scan_direction,
                    "seen:",
                    seen_landmarks
                )

                # Drej
                if scan_timer < SCAN_ROTATE_TIME:

                    if scan_direction > 0:
                        robot_controller.go_diff_calibrated(-1, 1)
                    else:
                        robot_controller.go_diff_calibrated(1, -1)

                # Pause
                elif scan_timer < SCAN_ROTATE_TIME + SCAN_PAUSE_TIME:

                    robot_controller.stop()

                # Ny rotationsperiode
                else:

                    scan_timer = 0.0
                    scan_direction *= -1

                    robot_controller.stop()

            else:

                if velocity > 0 and abs(angular_velocity) < 0.05:

                    robot_controller.go_diff_calibrated(1, 1)

                elif velocity > 0 and angular_velocity > 0:

                    robot_controller.go_diff_calibrated(0, 1)

                elif velocity > 0 and angular_velocity < 0:

                    robot_controller.go_diff_calibrated(1, 0)

                elif angular_velocity > 0:

                    robot_controller.go_diff_calibrated(-1, 1)

                elif angular_velocity < 0:

                    robot_controller.go_diff_calibrated(1, -1)

                else:

                    robot_controller.stop()

        # if isRunningOnArlo():
        #             if velocity > 0 and abs(angular_velocity) < 0.05:
        #                 robot_controller.go_diff_scan(1, 1)
        #             elif velocity > 0 and angular_velocity > 0:
        #                 robot_controller.go_diff_scan(0.5, 1)   # Blødt sving mod venstre under fremkørsel
        #             elif velocity > 0 and angular_velocity < 0:
        #                 robot_controller.go_diff_scan(1, 0.5)   # Blødt sving mod højre under fremkørsel
        #             elif angular_velocity > 0:
        #                 robot_controller.go_diff_scan(-1, 1)   # Roter til venstre på stedet
        #             elif angular_velocity < 0:
        #                 robot_controller.go_diff_scan(1, -1)   # Roter til højre på stedet
        #             else:
        #                 robot_controller.stop()
    
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

