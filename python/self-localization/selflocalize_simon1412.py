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

LANDMARK_DIST = 100.0
landmarkIDs = [2, 4]
landmarks = {
    2: (0.0, 0.0),
    4: (LANDMARK_DIST, 0.0)
}
landmark_colors = [CRED, CGREEN] 
target_x, target_y = LANDMARK_DIST / 2.0, 0.0

# Robotten styres med go_diff_calibrated, hvor hjulene er kalibreret til at køre ligeud.
# XXX: V_CALIB og W_CALIB SKAL MÅLES med go_diff_calibrated:
#  - kør ligeud i fx 3 s og mål afstanden -> V_CALIB = afstand / tid  [cm/s]
#  - drej i fx 4 s og mål vinklen        -> W_CALIB = vinkel / tid   [rad/s]
# Værdierne herunder er beregnede (ca. 31 cm/s og 90 grader/s), ikke målte.
WHEEL_2_WHEEL_DIST = 0.381

meters_pr_sec = 2 * PI * WHEEL_2_WHEEL_DIST / 8 * (3.14/3)
degrees_pr_second = 360 / 4

V_CALIB = meters_pr_sec * 100            # cm/s
W_CALIB = np.deg2rad(degrees_pr_second)  # rad/s

# Konvergens: partikelskyen skal være så samlet, før vi stoler på estimatet og kører mod målet
# Kravet behøver ikke være stramt: vi kører højst MAX_DRIVE_STEP ad gangen og observerer igen,
# så estimatet bliver bedre jo tættere vi kommer på landmarks.
POS_STD_THRESHOLD = 25.0     # cm (standardafvigelse i x og y)
HEADING_CONCENTRATION = 0.9  # længde af middel-retningsvektoren (1 = alle peger samme vej)

# Hvis skyen bliver mere spredt end dette under kørsel mod målet, går vi tilbage til SCAN
LOST_POS_STD_THRESHOLD = 50.0     # cm
LOST_HEADING_CONCENTRATION = 0.7

# Støj der tilføjes efter resampling, så partiklerne ikke kollapser til få kopier
RESAMPLE_POS_NOISE = 2.0     # cm
RESAMPLE_ANGLE_NOISE = 0.02  # rad

# Kameraet sidder foran robottens centrum. Partiklerne beskriver robottens centrum, så målingerne flyttes dertil.
# XXX: MÅL afstanden fra midten af hjulakslen til kameraet og skriv den her.
CAMERA_OFFSET = 0.0  # cm

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

    offsetX = 100
    offsetY = 250

    ymax = world.shape[0]

    world[:] = CWHITE

    max_weight = 0
    for particle in particles:
        max_weight = max(max_weight, particle.getWeight())

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


def particle_spread(particles):
    """Returnerer (std_x [cm], std_y [cm], heading_concentration) for partikelskyen."""
    xs = np.array([p.getX() for p in particles])
    ys = np.array([p.getY() for p in particles])
    thetas = np.array([p.getTheta() for p in particles])

    heading_concentration = np.hypot(np.mean(np.cos(thetas)), np.mean(np.sin(thetas)))

    return np.std(xs), np.std(ys), heading_concentration


def is_converged(particles):
    """Returnerer True når partikelskyen er samlet nok til at gennemsnittet er et brugbart estimat."""
    std_x, std_y, heading_concentration = particle_spread(particles)

    return (std_x < POS_STD_THRESHOLD
            and std_y < POS_STD_THRESHOLD
            and heading_concentration > HEADING_CONCENTRATION)


def is_lost(particles):
    """Returnerer True når skyen er blevet så spredt (fx efter lang kørsel uden landmarks),
    at vi bør scanne igen. Grænsen er højere end konvergens-grænsen, så vi ikke skifter frem og tilbage."""
    std_x, std_y, heading_concentration = particle_spread(particles)

    return (std_x > LOST_POS_STD_THRESHOLD
            or std_y > LOST_POS_STD_THRESHOLD
            or heading_concentration < LOST_HEADING_CONCENTRATION)

def mcl_step(particles, u_t, z_t, delta_t):
    """
    Funktion der udfører MCL-skridt: Prediction, Correction og Resampling.
    u_t er (velocity [cm/s], angular_velocity [rad/s]) for den bevægelse robotten har udført i delta_t sekunder.
    Vi opdaterer partikel vægt og resampler hvis vi har landmark i frame.
    """
    velocity, angular_velocity = u_t
    objectIDs, dists, angles = z_t

    if delta_t > 0.0 and (velocity != 0.0 or angular_velocity != 0.0):
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


def camera_to_robot_center(dist, angle):
    """Omregner (afstand, vinkel) målt fra kameraet til (afstand, vinkel) fra robottens centrum."""
    x = CAMERA_OFFSET + dist * np.cos(angle)
    y = dist * np.sin(angle)
    return np.hypot(x, y), np.arctan2(y, x)


TARGET_TOLERANCE = 15.0           # cm
ANGLE_TOLERANCE = 0.15            # rad ≈ 8.6°
SCAN_STEP_MIN = np.deg2rad(25)    # rad, hvor meget vi drejer pr. scan-skridt (tilfældigt i intervallet,
SCAN_STEP_MAX = np.deg2rad(35)    #      så vi ikke kigger i præcis de samme retninger hver omgang)
MAX_DRIVE_STEP = 40.0             # cm, længste stykke vi kører før vi stopper og observerer igen
SETTLE_TIME = 0.4                 # s, pause efter bevægelse så kamerabilledet ikke er sløret

def autonomous_controller(
    drive_state,
    estimated_pose,
    target,
    seen_landmarks,
    converged,
    lost
):
    """
    Skridtvis controller. Returnerer (command, amount, next_state), hvor
    command er "ROTATE" (amount i rad, positiv = venstre), "DRIVE" (amount i cm) eller "STOP".
    """
    x = estimated_pose.getX()
    y = estimated_pose.getY()
    theta = estimated_pose.getTheta()

    target_x, target_y = target

    dx = target_x - x
    dy = target_y - y

    distance = np.hypot(dx, dy)

    target_angle = np.arctan2(dy, dx)

    angle_error = wrap_angle(target_angle - theta)

    # Har vi mistet positionen under kørslen, scanner vi igen
    if drive_state == "GO_TO_TARGET" and lost:
        drive_state = "SCAN"

    #Vi scanner indtil begge landmarks er blevet set OG partiklerne har konvergeret.
    #Før det er estimatet bare gennemsnittet af en spredt sky og kan ikke bruges.
    if drive_state == "SCAN":

        if len(seen_landmarks) >= 2 and converged:
            drive_state = "GO_TO_TARGET"
        else:
            return "ROTATE", np.random.uniform(SCAN_STEP_MIN, SCAN_STEP_MAX), "SCAN"

    if drive_state == "GO_TO_TARGET":

        #Hvis vi er indenfor mål tolerancen, dvs. meget tæt på målet stopper vi
        if distance < TARGET_TOLERANCE:
            return "STOP", 0.0, "STOP"

        # Drej først præcis mod målet, kør derefter et begrænset stykke og observér igen
        if abs(angle_error) > ANGLE_TOLERANCE:
            return "ROTATE", angle_error, "GO_TO_TARGET"

        return "DRIVE", min(distance, MAX_DRIVE_STEP), "GO_TO_TARGET"

    return "STOP", 0.0, "STOP"


def execute_robot_command(robot, command, amount):
    """
    Udfører én blokerende robotkommando: bevæger sig i den udregnede tid, stopper og venter kort.

    Returnerer den bevægelse motion model skal bruge:
        velocity         [cm/s]
        angular_velocity [rad/s]
        duration         [s]
    """
    if robot is None:
        return 0.0, 0.0, 0.0

    if command == "ROTATE":
        duration = abs(amount) / W_CALIB
        if amount > 0:
            robot.go_diff_calibrated(-1, 1)
            angular_velocity = W_CALIB
        else:
            robot.go_diff_calibrated(1, -1)
            angular_velocity = -W_CALIB
        time.sleep(duration)
        robot.stop()
        time.sleep(SETTLE_TIME)
        return 0.0, angular_velocity, duration

    if command == "DRIVE":
        duration = amount / V_CALIB
        robot.go_diff_calibrated(1, 1)
        time.sleep(duration)
        robot.stop()
        time.sleep(SETTLE_TIME)
        return V_CALIB, 0.0, duration

    # STOP og ukendte kommandoer
    robot.stop()
    return 0.0, 0.0, 0.0


# Main
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

    drive_state = "SCAN"

    # Den bevægelse robotten udførte i sidste iteration (bruges af motion model)
    last_velocity = 0.0
    last_angular_velocity = 0.0
    last_duration = 0.0

    # Initialize the robot
    robot_controller = robot_module.Robot() if isRunningOnArlo() else None

    # Allocate space for world map
    world = np.zeros((500,500,3), dtype=np.uint8)

    print("Opening and initializing camera")
    if isRunningOnArlo():
        #cam = camera.Camera(0, robottype='arlo', useCaptureThread=True)
        cam = camera.Camera(0, robottype='arlo', useCaptureThread=False)
    else:
        cam = camera.Camera(0, robottype='macbookpro', useCaptureThread=True)
        #cam = camera.Camera(1, robottype='macbookpro', useCaptureThread=False)

    seen_landmarks = set()#Vi gemmer de landmark-ID'er vi har set

    while True:

        # Robotten står stille nu. Det første billede smides væk,
        # så vi ikke bruger et billede taget mens robotten stadig bevægede sig.
        cam.get_next_frame()
        colour = cam.get_next_frame()
        detected_objectIDs, detected_dists, detected_angles = cam.detect_aruco_objects(colour)

        objectIDs=[]
        dists=[]
        angles=[]

        if detected_objectIDs is not None:
            VALID_IDS = set(landmarkIDs)
            observations={}
            #hvis vi har dubletter af samme id, vælger vi tætteste distance
            for ID, dist, angle in zip(detected_objectIDs, detected_dists, detected_angles):
                ID = int(ID)
                if ID not in VALID_IDS:
                    continue

                if ID not in observations or dist < observations[ID][0]:
                    observations[ID] = (dist, angle)

            for ID, (measured_dist, measured_angle) in observations.items():
                measured_dist, measured_angle = camera_to_robot_center(measured_dist, measured_angle)
                objectIDs.append(ID)
                dists.append(measured_dist)
                angles.append(measured_angle)

                seen_landmarks.add(ID)

        # MCL-KALD: bevægelsen fra sidste kommando + observationerne fra dette billede
        u_t = (last_velocity, last_angular_velocity)
        z_t = (objectIDs, dists, angles)
        particles = mcl_step(particles, u_t, z_t, last_duration)

        # Tegn detekterede ArUco-mærker på kamerabilledet hvis fundet
        if objectIDs:
            cam.draw_aruco_objects(colour)

        # Beregn robottens estimerede position efter MCL-opdateringen
        est_pose = particle.estimate_pose(particles)
        converged = is_converged(particles)
        lost = is_lost(particles)

        command, amount, drive_state = autonomous_controller(
                drive_state,
                est_pose,
                (target_x, target_y),
                seen_landmarks,
                converged,
                lost
            )

        std_x, std_y, heading_concentration = particle_spread(particles)
        print("observations", list(zip(objectIDs, np.round(dists, 1), np.round(angles, 3))))
        print("seen", seen_landmarks, "converged", converged, "lost", lost)
        print("pose x=%.1f y=%.1f theta=%.1f deg | std_x=%.1f std_y=%.1f heading=%.2f"
              % (est_pose.getX(), est_pose.getY(), np.rad2deg(wrap_angle(est_pose.getTheta())),
                 std_x, std_y, heading_concentration))
        print("state", drive_state, "command", command, round(float(amount), 3))

        if showGUI:
            draw_world(est_pose, particles, world)
            cv2.imshow(WIN_RF1, colour)
            cv2.imshow(WIN_World, world)
            if cv2.waitKey(10) == ord('q'): # Quit
                break

        last_velocity, last_angular_velocity, last_duration = execute_robot_command(
            robot_controller, command, amount
        )

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
