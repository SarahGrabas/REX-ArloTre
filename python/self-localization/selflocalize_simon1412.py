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
# XXX: MÅL afstanden mellem jeres to markers på gulvet og skriv den her. ID 1 skal stå i origo.
LANDMARK_DIST = 100.0  # [cm] afstand mellem landmark 1 og landmark 11
landmarkIDs = [2, 4]
landmarks = {
    2: (0.0, 0.0),  # Coordinates for landmark 1
    4: (LANDMARK_DIST, 0.0)  # Coordinates for landmark 2
}
landmark_colors = [CRED, CGREEN] # Colors used when drawing the landmarks
target_x, target_y = LANDMARK_DIST / 2.0, 0.0 #MÅL: midten mellem landmarks

# Motorkraft til go_diff (skal være 0 eller i [40;127])
DRIVE_POWER = 64
TURN_POWER = 64

# XXX: V_CALIB og W_CALIB SKAL MÅLES med DRIVE_POWER/TURN_POWER ovenfor:
#  - kør ligeud i fx 3 s og mål afstanden -> V_CALIB = afstand / tid  [cm/s]
#  - drej i fx 3 s og mål vinklen        -> W_CALIB = vinkel / tid   [rad/s]
# Værdierne herunder er gamle beregnede værdier (ca. 31 cm/s og 90 grader/s) og passer næppe.
WHEEL_2_WHEEL_DIST = 0.381

meters_pr_sec = 2 * PI * WHEEL_2_WHEEL_DIST / 8 * (3.14/3)
degrees_pr_second = 360 / 4

V_CALIB = meters_pr_sec * 100
W_CALIB = np.deg2rad(degrees_pr_second)

# Konvergens: partikelskyen skal være så samlet, før vi stoler på estimatet og kører mod målet
POS_STD_THRESHOLD = 15.0     # cm (standardafvigelse i x og y)
HEADING_CONCENTRATION = 0.9  # længde af middel-retningsvektoren (1 = alle peger samme vej)

# Støj der tilføjes efter resampling, så partiklerne ikke kollapser til få kopier
RESAMPLE_POS_NOISE = 2.0     # cm
RESAMPLE_ANGLE_NOISE = 0.02  # rad

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
    sigma_angle=0.20
    
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
                1 / (np.sqrt(2 * np.pi) * sigma_angle)) * np.exp(-0.5 * (wrap_angle(final_theta_particle - angles[i]) / sigma_angle)**2)
            
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
    """We use current weight for particles given in the update particle function.
    We randomly select same number of particles as the set we sample from.
    We update the particle list by replacing previous particles with new particles, and we set the weight for each particle to be equal"""
    weights = np.asarray([p.getWeight() for p in particles], dtype=float)
    selected = np.random.choice(len(particles), size=len(particles), replace=True, p=weights)
    new_particles = [
        particle.Particle(
            particles[index].getX(),
            particles[index].getY(),
            particles[index].getTheta(),
            1.0 / len(particles),
        )
        for index in selected
    ]

    # Lidt støj efter resampling, så skyen ikke kollapser til få identiske partikler (particle depletion)
    particle.add_uncertainty(new_particles, RESAMPLE_POS_NOISE, RESAMPLE_ANGLE_NOISE)

    return new_particles


def is_converged(particles):
    """Returnerer True når partikelskyen er samlet nok til at gennemsnittet er et brugbart estimat."""
    xs = np.array([p.getX() for p in particles])
    ys = np.array([p.getY() for p in particles])
    thetas = np.array([p.getTheta() for p in particles])

    heading_concentration = np.hypot(np.mean(np.cos(thetas)), np.mean(np.sin(thetas)))

    return (np.std(xs) < POS_STD_THRESHOLD
            and np.std(ys) < POS_STD_THRESHOLD
            and heading_concentration > HEADING_CONCENTRATION)

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
    seen_landmarks,
    converged
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

    #Vi scanner indtil begge landmarks er blevet set OG partiklerne har konvergeret.
    #Før det er estimatet bare gennemsnittet af en spredt sky og kan ikke bruges.
    if drive_state == "SCAN":

        if len(seen_landmarks) >= 2 and converged:
            return "ROTATE_TO_TARGET", "ROTATE_TO_TARGET"

        return "SCAN", "SCAN"

    #Hvis vi er indenfor mål tolerancen, dvs. meget tæt på målet stopper vi
    if distance < TARGET_TOLERANCE:
        return "STOP", "STOP"


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

last_sent_command = None

def execute_robot_command(robot, command):
    """
    Udfører én ikke-blokerende robotkommando.
    Kommandoen sendes kun til Arduinoen, når den er ændret, så den ikke oversvømmes.

    Returnerer den faktiske bevægelse:
        velocity       [cm/s]
        angular_velocity [rad/s]

    som robotten forventes at have under kommandoen.
    """
    global last_sent_command
    send = command != last_sent_command

    if command == "DRIVE_FORWARD":

        if send:
            robot.go_diff(DRIVE_POWER, DRIVE_POWER, 1, 1)
        last_sent_command = command

        return V_CALIB, 0.0

    elif command == "ROTATE_LEFT":

        if send:
            robot.go_diff(TURN_POWER, TURN_POWER, 0, 1)
        last_sent_command = command

        return 0.0, W_CALIB

    elif command == "ROTATE_RIGHT":

        if send:
            robot.go_diff(TURN_POWER, TURN_POWER, 1, 0)
        last_sent_command = command

        return 0.0, -W_CALIB

    # STOP og ukendte kommandoer
    if last_sent_command != "STOP":
        robot.stop()
    last_sent_command = "STOP"

    return 0.0, 0.0


# Main program #
cam = None
robot_controller = None
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
    velocity = 0.0
    angular_velocity = 0.0
    drive_state = "SCAN"

    # Den kommando robotten faktisk kørte med
    last_mcl_velocity = 0.0
    last_mcl_angular_velocity = 0.0

    # Scan state
    scan_direction = 1
    scan_timer = 0.0

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


    # velocity = 0.0
    # angular_velocity = 0.0
    # drive_state = "SCAN"
    # last_time = timer()
    
    # scan_direction = 1
    # scan_timer = 0.0

    import random

    SCAN_ROTATE_TIME = random.uniform(1.0, 2.3) #Når vi er i scan mode vælger vi tilfældig vinkel at dreje
    SCAN_PAUSE_TIME = 0.50

    seen_landmarks = set()#Vi gemmer vores observationer her, men altid den tætteste dublet den ser.

    # Starttid sættes først her, så kamera-initialiseringen ikke tæller med i første delta_t
    last_time = time.time()

    while True:

        action = cv2.waitKey(10)
        if action == ord('q'): # Quit
            break

        current_time = time.time()
        delta_t = current_time - last_time #hvor lang tid der er gået siden sidste MCL opdatering og ikke køretiden
        last_time = current_time

        # Beskyt mod mærkelige tidsværdier. Grænsen må ikke være lavere end en langsom iteration
        # (kamerabillede + MCL på Pi'en), ellers mister motion-modellen bevægelse.
        delta_t = max(0.0, min(delta_t, 2.0))

        # Hent kamerabillede og detekter ArUco-mærker (giver z_t)
        colour = cam.get_next_frame()
        detected_objectIDs, detected_dists, detected_angles = cam.detect_aruco_objects(colour)
        if not isinstance(detected_objectIDs, type(None)):
            # List detected objects
            for i in range(len(detected_objectIDs)):
                print("Object ID = ", detected_objectIDs[i], ", Distance = ", detected_dists[i], ", angle = ", detected_angles[i])
        
        objectIDs=[]    
        dists=[]
        angles=[]    
                    
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
        u_t = (last_mcl_velocity, last_mcl_angular_velocity) #vi bruger last velocity så det kun er bevægelse i det loop der svarer til delta t tiden.
                                                            #så når der ikke er nogle bvægelse kaldes motion model ikke.
        z_t = (objectIDs, dists, angles)
        particles = mcl_step(particles, u_t, z_t, delta_t)
        

        # Tegn detekterede ArUco-mærker på kamerabilledet hvis fundet
        if objectIDs:
            cam.draw_aruco_objects(colour)

        # Beregn robottens estimerede position efter MCL-opdateringen
        est_pose = particle.estimate_pose(particles)
        converged = is_converged(particles)


        command, drive_state = autonomous_controller(
                drive_state,
                est_pose,
                (target_x, target_y),
                seen_landmarks,
                converged
            )

        if isRunningOnArlo():

            if command == "SCAN":

                scan_timer += delta_t

                print(
                    "SCAN timer:",
                    round(scan_timer, 2),
                    "seen:",
                    seen_landmarks,
                    "converged:",
                    converged
                )

                # Vi drejer altid samme vej (scan_direction), så hele 360 grader bliver dækket.
                # Pausen giver skarpe billeder uden motion blur.
                if scan_timer < SCAN_ROTATE_TIME:

                    scan_command = "ROTATE_LEFT" if scan_direction > 0 else "ROTATE_RIGHT"
                    last_mcl_velocity, last_mcl_angular_velocity = execute_robot_command(robot_controller, scan_command)

                elif scan_timer < SCAN_ROTATE_TIME + SCAN_PAUSE_TIME:

                    last_mcl_velocity, last_mcl_angular_velocity = execute_robot_command(robot_controller, "STOP")

                else:

                    scan_timer = 0.0
                    SCAN_ROTATE_TIME = random.uniform(1.0, 2.3)

                    last_mcl_velocity, last_mcl_angular_velocity = execute_robot_command(robot_controller, "STOP")

            else:

                last_mcl_velocity, last_mcl_angular_velocity = execute_robot_command(robot_controller, command)

        #if showGUI:
            #draw_world(est_pose, particles, world)
            #cv2.imshow(WIN_RF1, colour)
            #cv2.imshow(WIN_World, world)

        #Vi stopper når controlleren har nået målet (først muligt efter konvergens)
        if drive_state == "STOP":
            print("Mål nået. Estimeret position:", est_pose.getX(), est_pose.getY())
            break


finally:
    # Make sure to clean up even if an exception occurred

    # Stop motorerne, så robotten ikke kører videre efter programmet er slut
    if robot_controller is not None:
        robot_controller.stop()

    # Close all windows
    cv2.destroyAllWindows()

    # Clean-up capture thread
    if cam is not None:
        cam.terminateCaptureThread()

