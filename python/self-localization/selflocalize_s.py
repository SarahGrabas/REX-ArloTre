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


def update_weights(new_particles,objectIDs,angles, dists):
    #Vi beregn weight ved distance og vinkler
    particle_weights =[]
    sigma_dist = 0.15
    sigma_angle=0.10
    for p in new_particles:
        x,y,theta =p
        
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
            enheds_theta=np.array(np.cos(theta), np.sin(theta))
            enheds_particle=np.array(diff_x, diff_y)/np.linalg.norm(np.array(diff_x, diff_y))
            
            theta_particle=np.arccos(np.dot(enheds_theta,enheds_particle) )
            
            #Vi skal finde ud af om theta_p er negativ eller positiv, dvs om den ligger på højre eller venstre side af enheds theta
            enheds_theta_hat=np.array(np.sin(theta),-np.cos(theta))
            
            final_theta_particle=np.sign(np.dot(enheds_particle, enheds_theta_hat))*theta_particle
            
            land_angle = (
                1 / (np.sqrt(2 * np.pi) * sigma_angle)) * np.exp(-0.5 * ((final_theta_particle - angles[i]) / sigma_angle)**2)
            
            land_weight=land_dist*land_angle

        #Hvis der er flere landmarks, ganges deres sandsynligheder sammen
            weight *= land_weight
        
        particle_weights.append(weight)    
        
        particle_weights = np.array(particle_weights)

        total_weight = np.sum(particle_weights)

        if total_weight > 0:
            particle_weights = particle_weights / total_weight
        
        for p, p_weight in zip(particles,particle_weights):
            p.setWeight(p_weight)


def resample_particles(particles):
    weights = np.asarray([p.getWeight() for p in particles], dtype=float)
    #weights /= np.sum(weights)
    selected = np.random.choice(len(particles), size=len(particles), replace=True, p=weights)
    return [
        particle.Particle(
            particles[index].getX(),
            particles[index].getY(),
            particles[index].getTheta(),
            1.0 / len(particles), #nulstiller vægte
        )
        for index in selected
    ]


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

    last_time = timer()
    while True:

        # Move the robot according to user input (only for testing)
        action = cv2.waitKey(10)
        if action == ord('q'): # Quit
            break
    
        # if not isRunningOnArlo():
        #     if action == ord('w'): # Forward
        #         velocity += 4.0
        #     elif action == ord('x'): # Backwards
        #         velocity -= 4.0
        #     elif action == ord('s'): # Stop
        #         velocity = 0.0
        #         angular_velocity = 0.0
        #     elif action == ord('a'): # Left
        #         angular_velocity += 0.2
        #     elif action == ord('d'): # Right
        #         angular_velocity -= 0.2

        # Use motor controls to update particles
        # XXX: Make the robot drive
        # XXX: You do this

        # Simon
        V_CALIB = 40.0   # Den kører 40 cm pr. sekund ca. Fremadrettet hastighed ved go_diff_calibrated(1, 1)
        W_CALIB = 1.9234  # I radianer/pr. sekund. Rotationshastighed (~110,2 degree/s) ved go_diff_calibrated(-1, 1)

        if action == ord('w'):     # Fremad
            velocity += V_CALIB
        elif action == ord('x'):   # Bagud
            velocity += -V_CALIB
        elif action == ord('a'):   # Venstre-rotation
            angular_velocity += W_CALIB
        elif action == ord('d'):   # Højre-rotation
            angular_velocity += -W_CALIB
        elif action == ord('s'):   # Stop
            velocity = 0.0
            angular_velocity = 0.0

        #Vi skal være opmærksom på at der ikke er sleep i vores funktioner, da den bare skal blive ved med at køre, da vi måler tid den kører
        if isRunningOnArlo():
            if velocity > 0:
                robot_controller.go_diff_calibrated(1, 1)
            elif velocity < 0:
                robot_controller.go_diff_calibrated(-1, -1)
            elif angular_velocity > 0:
                robot_controller.go_diff_calibrated(-1, 1)
            elif angular_velocity < 0:
                robot_controller.go_diff_calibrated(1, -1)
            else:
                robot_controller.stop()

        current_time = timer()
        delta_t = current_time - last_time
        last_time = current_time

        #Opdater partikler efter robot har kørt
        if velocity != 0.0 or angular_velocity != 0.0:
            new_particles=sample_motion_model_velocity(particles, velocity, angular_velocity, delta_t)


        # Fetch next frame
        colour = cam.get_next_frame()
        
        # Detect objects
        objectIDs, dists, angles = cam.detect_aruco_objects(colour)
        if not isinstance(objectIDs, type(None)):
            # List detected objects
            for i in range(len(objectIDs)):
                print("Object ID = ", objectIDs[i], ", Distance = ", dists[i], ", angle = ", angles[i])
            
        
        update_weights(new_particles,objectIDs,angles, dists)
        
        resample_particles(new_particles)

                
        #Nu skal vi så resample fra particle weights


            # Draw detected objects
            #cam.draw_aruco_objects(colour)
        est_pose = particle.estimate_pose(particles) # The estimate of the robots current pose

        if showGUI:
            # Draw map
            draw_world(est_pose, particles, world)
    
            # Show frame
            cv2.imshow(WIN_RF1, colour)

            # Show world
            cv2.imshow(WIN_World, world)
    
  
finally: 
    # Make sure to clean up even if an exception occurred
    
    # Close all windows
    cv2.destroyAllWindows()

    # Clean-up capture thread
    if cam is not None:
        cam.terminateCaptureThread()

