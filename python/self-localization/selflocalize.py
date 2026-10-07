import cv2
import particle
import camera
import numpy as np
from timeit import default_timer as timer
import sys
from motion_model import sample_motion_model_velocity


# Flags
showGUI = True  # Whether or not to open GUI windows
onRobot = True  # Whether or not we are running on the Arlo robot


def isRunningOnArlo():
    """Return True if we are running on Arlo, otherwise False.
      You can use this flag to switch the code from running on you laptop to Arlo - you need to do the programming here!
    """
    return onRobot


robot_module = None
if isRunningOnArlo():
    # XXX: You need to change this path to point to where your robot.py file is located
    sys.path.append("../../../../Arlo/python")
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
landmarkIDs = [1, 2]
landmarks = {
    1: (0.0, 0.0),  # Coordinates for landmark 1
    2: (300.0, 0.0)  # Coordinates for landmark 2
}
landmark_colors = [CRED, CGREEN] # Colors used when drawing the landmarks
range_sigma = 10.0
bearing_sigma = 0.15

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


def update_particle_weights(particles, object_ids, distances, angles):
    observations = [
        (int(object_id), distance, angle)
        for object_id, distance, angle in zip(object_ids, distances, angles)
        if int(object_id) in landmarks
    ]
    if not observations:
        return False

    log_weights = []
    for p in particles:
        log_weight = np.log(max(p.getWeight(), 1e-300))
        for object_id, measured_range, measured_bearing in observations:
            landmark_x, landmark_y = landmarks[object_id]
            dx = landmark_x - p.getX()
            dy = landmark_y - p.getY()
            predicted_range = np.hypot(dx, dy)
            predicted_bearing = wrap_angle(np.arctan2(dy, dx) - p.getTheta())

            range_error = (measured_range - predicted_range) / range_sigma
            bearing_error = wrap_angle(measured_bearing - predicted_bearing) / bearing_sigma
            log_weight -= 0.5 * (range_error**2 + bearing_error**2)
        log_weights.append(log_weight)

    weights = np.exp(np.asarray(log_weights) - np.max(log_weights))
    weights /= np.sum(weights)
    for p, weight in zip(particles, weights):
        p.setWeight(weight)
    return True


def resample_particles(particles):
    weights = np.asarray([p.getWeight() for p in particles], dtype=float)
    weights /= np.sum(weights)
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
    if not isinstance(objectIDs, type(None)):
        if update_particle_weights(particles, objectIDs, dists, angles):
            particles = resample_particles(particles)

    return particles

# udkast til en tilstandsmaskine ift. at finde midterpunktet mellem to landmarks
def autonomous_controller(est_pose, objectIDs, drive_state):
    """
    Beregner motorkommandoer (velocity, angular_velocity) baseret på 
    MCL-estimatet (est_pose) og den aktive tilstand.
    """
    target_x, target_y = 150.0, 0.0  # Mål: midten mellem landemærkerne (0,0) og (300,0)
    
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
        angular_velocity = W_CALIB
        velocity = 0.0
        
        if not isinstance(objectIDs, type(None)) and len(set(objectIDs)) >= 2:
            drive_state = "ROTATE_TO_TARGET"

    elif drive_state == "ROTATE_TO_TARGET":
        if abs(angle_error) > 0.08:  # Ca. 4.5 grader
            angular_velocity = np.sign(angle_error) * W_CALIB 
            velocity = 0.0
        else:
            drive_state = "DRIVE_TO_TARGET"

    elif drive_state == "DRIVE_TO_TARGET":
        if dist_to_target > 10.0:
            velocity = V_CALIB
            # P-regulator der holder kursen mod målet under kørsel
            angular_velocity = np.clip(0.8 * angle_error, -W_CALIB, W_CALIB)
        else:
            drive_state = "STOP"

    elif drive_state == "STOP":
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
    world = np.zeros((500,500,3), dtype=np.uint8)

    # Draw map
    draw_world(est_pose, particles, world)

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
    while True:

        action = cv2.waitKey(10)
        if action == ord('q'): # Quit
            break

        current_time = timer()
        delta_t = current_time - last_time
        last_time = current_time

        # Hent kamerabillede og detekter ArUco-mærker (giver z_t)
        colour = cam.get_next_frame()
        objectIDs, dists, angles = cam.detect_aruco_objects(colour)

        # MCL-KALD
        u_t = (velocity, angular_velocity)
        z_t = (objectIDs, dists, angles)
        particles = mcl_step(particles, u_t, z_t, delta_t)

        # Tegn detekterede ArUco-mærker på kamerabilledet hvis fundet
        if not isinstance(objectIDs, type(None)):
            cam.draw_aruco_objects(colour)

        # Beregn robottens estimerede position efter MCL-opdateringen
        est_pose = particle.estimate_pose(particles)

        velocity, angular_velocity, drive_state = autonomous_controller(
        est_pose, objectIDs, drive_state)

        if isRunningOnArlo():
                    if velocity > 0 and abs(angular_velocity) < 0.05:
                        robot_controller.go_diff_calibrated(1, 1)
                    elif velocity > 0 and angular_velocity > 0:
                        robot_controller.go_diff_calibrated(0.5, 1)   # Blødt sving mod venstre under fremkørsel
                    elif velocity > 0 and angular_velocity < 0:
                        robot_controller.go_diff_calibrated(1, 0.5)   # Blødt sving mod højre under fremkørsel
                    elif angular_velocity > 0:
                        robot_controller.go_diff_calibrated(-1, 1)   # Roter til venstre på stedet
                    elif angular_velocity < 0:
                        robot_controller.go_diff_calibrated(1, -1)   # Roter til højre på stedet
                    else:
                        robot_controller.stop()
    
        if showGUI:
            draw_world(est_pose, particles, world)
            cv2.imshow(WIN_RF1, colour)
            cv2.imshow(WIN_World, world)
  
finally: 
    # Make sure to clean up even if an exception occurred
    
    # Close all windows
    cv2.destroyAllWindows()

    # Clean-up capture thread
    if cam is not None:
        cam.terminateCaptureThread()

