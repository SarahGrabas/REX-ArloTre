##### TYPING #####

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray

##### LIBARIES #####

from time import sleep
import numpy as np
from math import sqrt


##### PROGRAM PARAMETERS #####

VERBOSITY : int = 1 # 0 means no debug info, 1 means all, and higher values means less verbosity



##### CONSTANTS #####

from math import pi as PI

ARLO_RADIUS = 0.225 # in meters

WHEEL_2_WHEEL_DIST = 0.381 # center of wheel to center of wheel, not edge to edge

MARKER_SIZE = 0.146 # markørstørrelse i meter på landmarkbox

FOCAL_LENGTH = 1288.9 # Camera focal length from Exercise 3.1

RESOLUTION = (1640, 1232) # x, y in pixels




##### PROGRAM BODY #####

# in case verbosity is false, we want debug calls to not throw an unknown expression error
def debug(*args, **kwargs): pass

# if verbosity is true, a debug function which prints to out in a standard way, is defined
if VERBOSITY:
    # defines elapsed_time function for getting time since program start, used by the debug print function
    from time import perf_counter

    _START_TIME = perf_counter()

    def _elapsed_time():
        elapsed_ms = int((perf_counter() - _START_TIME) * 1000)

        minutes = elapsed_ms // 60_000
        seconds = (elapsed_ms // 1_000) % 60
        milliseconds = elapsed_ms % 1_000

        return f"{minutes:02}:{seconds:02}:{milliseconds:03}"

    # defines the debug print function
    def debug(*values: object, verbosity:int=9):
        """A debug print function used for printing formatted debug information to out, if verbosity is true.\\
        The higher `verbosity` is set (at least 1), the higher the value, the more important the debug info is."""
        if verbosity >= VERBOSITY:
            print(
                f"[{_elapsed_time()}] ({verbosity}) ",
                *values,
                file = None 
            )
        else:
            assert verbosity >= 1, f"Error: Expected verbosity to be positive, but recieved {verbosity}"






class Port(object):
    """
    A class for establishing a port connection between the between the Rasberry Pi and Arduino.\\
    Used for interacting with the moters and sonar sensors.

    Not thread safe, but can be run it a single thread.
    """
    def __init__(self, port = '/dev/ttyACM0'):
        from serial import Serial # type: ignore

        debug("Waiting for serial port connection ...", verbosity=2)

        self.port = port
        
        self.serialRead = Serial(self.port,9600, timeout=None) # No timeout, wait forever or until data is received

        # Wait if serial port is not open yet
        while not self.serialRead.isOpen():
            sleep(0.25)

        sleep(2)

        debug("Connected to seriel port!", verbosity=2)
                
    def send_command(self, cmd:str, sleep_ms=0.0):
        """Sends a command to the Arduino robot controller"""
        debug(f"send_command({self}, {cmd}, {sleep_ms})", verbosity=1)

        self.serialRead.write(cmd.encode('ascii'))
        sleep(sleep_ms)
        str_val=self.serialRead.readline()
        return str_val



class Motor:
    def __init__(self, port:Port):
        self._port = port
        debug("Initialized moters!", verbosity=1)
    
    @staticmethod
    def _power_checker(power:int) -> bool:
        """Checks if a power value is in the set {0, [40;127]}.
           This is an internal utility function."""
        return (power == 0) or (power >=40 and power <=127) 

    def _move(self, powerLeft, powerRight, dirLeft, dirRight):
        """Start left motor with motor power powerLeft (in {0, [40;127]} and the numbers must be integer) and direction dirLeft (0=reverse, 1=forward)
           and right motor with motor power powerRight (in {0, [40;127]} and the numbers must be integer) and direction dirRight (0=reverse, 1=forward).
        
           The Arlo robot may blow a fuse if you run the motors at less than 40 in motor power, therefore choose either 
           power = 0 or 40 < power <= 127.
           
           This does NOT use wheel encoders."""
        debug(f"_move({self}, {powerLeft}, {powerRight}, {dirLeft}, {dirRight})", verbosity=1)
        
        # checks for legallity of requisted speed
        assert self._power_checker(powerLeft) and self._power_checker(powerRight), "Error: Expected power in range {0, [40;127]}, but recieved "+f"{powerLeft, powerRight}"

        cmd = 'd' + str(int(powerLeft)) + ',' + str(int(powerRight)) + ',' + str(int(dirLeft)) + ',' + str(int(dirRight)) + '\n'
        return self._port.send_command(cmd)

    def stop(self):
        """Send a stop command to stop motors. Sets the motor power on both wheels to zero.
        
           This does NOT use wheel encoders."""
        debug("stop()", verbosity=3)
        cmd = 's\n'
        return self._port.send_command(cmd)
    
    def move(self, dirLeft:int, dirRight:int):
        """
        Like ``Robot.go_diff`` but calibrated for speeds in the range [50;60].\\
        ``dirLeft`` and ``dirRight`` can be ``[-1,0,1]``, where:
        - ``-1`` is backwards
        - ``0`` is still/none
        - ``1`` is forward.
        """
        debug(f"move({self}, {dirLeft}, {dirRight})", verbosity=2)
        assert dirLeft in (-1,0,1), f"Error: Expected dirLeft to be either -1,0 or 1, but recieved {dirLeft}"
        assert dirRight in (-1,0,1), f"Error: Expected dirRight to be either -1,0 or 1, but recieved {dirRight}"
        
        speedLeft = (0, 51, 51)[dirLeft]
        speedRight = (0, 57, 53)[dirRight]

        self._move(speedLeft, speedRight, 1 if dirLeft==1 else 0, 1 if dirRight==1 else 0)

    def drive(self, meters:float, forward=True, _sleep=True, stop_when_done=True):
        """
        Drives straight forward or backward at ~0.3 meters/second.\\
        Returns the seconds driven.
        """
        debug(f"drive({self}, {meters}, {forward}, {_sleep}, {stop_when_done})", verbosity=3)
        assert meters >= 0, f"Error: Expected positive distance, but recieved {meters}"

        meters_pr_sec = 2 * PI * WHEEL_2_WHEEL_DIST / 8   * 3.14/3 # last factor is correction
        
        self.move(2 * forward - 1, 2 * forward - 1)
        t = meters / meters_pr_sec
        if _sleep: sleep(t)
        if stop_when_done: self.stop()
        debug(f"Expected time slept = {t} seconds", verbosity=2)
        return t

    def rotate(self, degrees:float, _sleep=True, stop_when_done=True):
        """
        Rotates ``degrees`` inplace, at 90°/second, and stops.\\
        Negative rotation is right, positive rotation is left.
        Returns the seconds driven.
        """
        debug(f"rotate({self}, {degrees}, {_sleep}, {stop_when_done})", verbosity=3)

        degrees_pr_second = 360 / 4 # 4 instead of 8 is intentional
        correction_fraction_negative = 1.0
        correction_fraction_positive = 1.0
    
        if degrees < 0:
            self.move(1, -1)
            correction_fraction = correction_fraction_negative
        elif degrees > 0:
            self.move(-1, 1)
            correction_fraction = correction_fraction_positive
        else:
            return
        t = abs(degrees) / degrees_pr_second / correction_fraction
        if _sleep: sleep(t)
        if stop_when_done: self.stop()
        debug(f"Expected time slept = {t} seconds", verbosity=2)
        return t




class Sonar:
    def __init__(self, port:Port):
        self._port = port
        debug("Initialized sensors!", verbosity=1)

    def _read_sensor(self, sensorid):
        """Send a read sensor command with sensorid and return sensor value. 
           Will return -1, if error occurs."""
        debug(f"_read_sensor({self}, {sensorid})", verbosity=1)

        cmd = str(sensorid) + '\n'
        str_val = self._port.send_command(cmd)
        debug(f"Sonar output (str_val) = {str_val}", verbosity=1)
        assert len(str_val) > 0, "Error: Unhandled edge case! Recieved empty string as sonar response."

        debug("Sonar distance [mm] = "+str_val, verbosity=3)
        return int(str_val)
            
    def front(self):
        """Read the front sonar ping sensor and return the measured range in milimeters [mm]"""
        debug(f"front({self})", verbosity=2)
        return self._read_sensor(0)
        
    def back(self):
        """Read the back sonar ping sensor and return the measured range in milimeters [mm]"""
        debug(f"back({self})", verbosity=2)
        return self._read_sensor(1)
        
    def left(self):
        """Read the left sonar ping sensor and return the measured range in milimeters [mm]"""
        debug(f"left({self})", verbosity=2)
        return self._read_sensor(2)
        
    def right(self):
        """Read the right sonar ping sensor and return the measured range in milimeters [mm]"""
        debug(f"right({self})", verbosity=2)
        return self._read_sensor(3)



class Landmark:
    MAX_BOX_SIZES = {
        1  : 0.28,
        2  : 0.275,
        3  : 0.28,
        4  : 0.285,
        5  : 0.275,
        6  : 0.31,
        7  : 0.37,
        8  : 0.245,
        9  : 0.245,
        10 : 0.275,
        11 : 0.305,
    }

    def __init__(self, id:int, center:NDArray, radius:float, normal:NDArray, distance:float, angle:float):
        debug(f"Landmark({id}, {center}, {radius}, {normal}, {distance}, {angle})")
        self.id = id
        self.center = center
        self.radius = radius
        self.normal = normal
        self.distance = distance
        self.angle = angle



class Camera:
    CX, CY = RESOLUTION[0]/2, RESOLUTION[1]/2 # Camera center in pixels

    CAMERA_MATRIX = np.array([
        [FOCAL_LENGTH, 0, CX], 
        [0, FOCAL_LENGTH, CY], 
        [0, 0, 1]
    ], dtype=np.float32) # 3x3 matrix 

    DISTORTION_MATRIX = np.zeros((5, 1), dtype=np.float32) # Zero vector ; assumes no camera (lens) distortion

    ROBOT_TO_CAMERA = np.array([
        [1, 0, 0, 0],
        [0, 0, -1, 0.208],
        [0, 1, 0, -0.225],
        [0, 0, 0, 1]
    ])

    CAMERA_TO_ROBOT = np.linalg.inv(ROBOT_TO_CAMERA)
    
    # ---------------------------------------------------------------------------
    # Robot coords
    # +X : Right
    # +Y : Front
    # +Z : Up 

    # Camera coords
    # +X : Right
    # +Y : Down
    # +Z : Front
    # ---------------------------------------------------------------------------

    def __init__(self):
        # imports cv2 lib for later use
        global cv2
        import cv2 # type: ignore

        self.cam = self._init_camera()
        debug("Initialized camera!", verbosity=1)

    @staticmethod
    def _init_camera(wait=False):
        from os import environ
        environ["LIBCAMERA_LOG_LEVELS"] = "*:WARN"

        debug("Initializing camera ...", verbosity=1)
        import picamera2 # type: ignore

        cam = picamera2.Picamera2() #open camera
        config = cam.create_video_configuration({ # define camera config suitable for recording video
            "size": RESOLUTION, 
            "format": "RGB888"
        }) 

        cam.configure(config) #use config
        cam.start(show_preview=False) #start camera (turn on)

        if wait: sleep(1) #wait for camera to start
        return cam

    def close(self):
        """Stop camera and release its resources."""
        if self.cam is not None:
            self.cam.stop()
            self.cam.close()
            self.cam = None
            debug("Camera closed!", verbosity=1)

    def capture(self) -> NDArray:
        """
        Takes picture in RBG format and return array of shape (hight, width, rbg).\\
        Returns (ids, rvecs, tvecs).
        """
        debug("Capturing image!", verbosity=2)
        return self.cam.capture_array("main") #the capture array function captures next image from the stream
    
    @staticmethod
    def _face_normal(rvec) -> NDArray:
        """Given an rotation vector, returns the unit length normal vector for the marker face."""
        R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float))
        debug(f"Calculated normal vector = {R[:, 2]}")
        return R[:, 2]

    def _detect_markers(self) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Returns (ids, tvecs, rvecs, face_normals) in camera coordinates.
        """
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

        frame = self.capture()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
        debug(f"Detected ids = {ids}", verbosity=1)

        # No markers detected
        if ids is None:
            debug(f"Returning empty ids, tvecs, rvecs, face_normals", verbosity=1)
            empty = np.empty((0, 3), dtype=np.float64)
            return np.empty(0, dtype=int), empty, empty.copy(), empty.copy()

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            MARKER_SIZE,
            Camera.CAMERA_MATRIX,
            Camera.DISTORTION_MATRIX
        )
        
        debug(f"cv2 calculated ids, rvecs, tvecs = {ids, rvecs, tvecs}", verbosity=1)
        ids = ids.flatten()
        rvecs = rvecs.reshape(-1, 3)
        tvecs = tvecs.reshape(-1, 3)

        normals = np.array([self._face_normal(rvec) for rvec in rvecs])

        return ids, tvecs, rvecs, normals

    def detect_landmarks(self) -> tuple[Landmark, ...]:
        """
        Detects landmarks visible to the camera.\\
        Returns a tuple of all visible landmarks, as ``Landmark`` objects
        """
        debug("Detecting landmarks!", verbosity=2)
        ids, tvecs, rvecs, normals = self._detect_markers()
        debug(f"Detected the following landmark ids = {ids}", verbosity=2)
        debug(f"ids, tvecs, rvecs, normals (in camera coords) = {ids, tvecs, rvecs, normals}", verbosity=1)

        # if no landmarks are found
        if len(ids) == 0:
            debug("No landmarks detected, returning empty tuple.", verbosity=1)
            return tuple()

        # Box centers in camera coordinates
        max_box_size = max(Landmark.MAX_BOX_SIZES.values())
        sizes = np.array([
            Landmark.MAX_BOX_SIZES.get(int(id), max_box_size)
            for id in ids
        ])
        debug(f"Landmark side lengths = {sizes}", verbosity=1)

        R = Camera.CAMERA_TO_ROBOT[:3, :3]
        t = Camera.CAMERA_TO_ROBOT[:3, 3]

        tvecs @= R.T
        normals @= R.T
        debug(f"Landmark tvecs in robot coords = {tvecs}", verbosity=1)
        debug(f"Landmark face normals in robot coords = {normals}", verbosity=1)

        centers = tvecs - normals * (sizes[:, None] / 2) + t
        debug(f"Landmark centers in robot coords = {centers}", verbosity=1)

        radiuses = [
            sqrt(Landmark.MAX_BOX_SIZES.get(int(id), max_box_size) ** 2 / 2)
            for id in ids
        ]
        debug(f"Landmark radiuses = {radiuses}", verbosity=1)

        # Distance from robot center to marker center
        distances = np.linalg.norm(centers[:, :2], axis=1)

        # Angle from robot center to marker center
        angles = np.degrees(
            np.arctan2(-centers[:, 0], centers[:, 1])
        )
        debug(f"Distance and angle from robot center to marker centers = {distances, angles}", verbosity=1)

        return tuple(
            Landmark(id, center, radius, normal, distance, angle)
            for id, center, radius, normal, distance, angle in zip(ids, centers, radiuses, normals, distances, angles)
        )



class Robot:
    def __init__(self):
        debug("Initializing robot ...")
        self._port = Port()
        self.moters = Motor(self._port)
        self.sonars = Sonar(self._port)
        self.camera = Camera()
        
        debug("Robot initilized!")

    def __enter__(self): # for with statement
        return self

    def __exit__(self, exc_type, exc_value, traceback): # for with statement
        self.close()
        return False

    def close(self):
        """Stop robot and release all resources."""
        debug("Stopping and terminating robot initialization ...")

        try: # stops moters
            sleep(0.05)
            self.moters.stop()
            sleep(0.1)
        finally:
            try:
                self.camera.close()
            finally:
                try:
                    self._port.send_command("k\n")
                finally:
                    self._port.serialRead.close()

        debug("Robot terminated!", verbosity=1)





with Robot() as arlo:
    arlo.camera.detect_landmarks()
    arlo.sonars.front()
    arlo.moters.rotate(90)
