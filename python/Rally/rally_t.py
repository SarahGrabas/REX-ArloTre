"""Rally-designforslag

Målversion: Python 3.9.2. Hardwaregrænseflader synkroniseret med rally.py.
Kør filen for et deterministisk planlægningseksempel UDEN hardware.
Dette er byggesten og en samlet beslutningsfunktion, ikke en færdig robotløkke.
Hardwareløkke, målt stopafstand, sonarmontage og kalibrering skal afprøves separat.

Globalt: origo nederst til venstre, x mod højre, y op, theta fra +x mod uret.
Kameraets Landmark: x mod højre, y frem, z op; angle beholdes i GRADER.
Nye navigationsdata: meter, RADIANER og time.monotonic()-sekunder.
Localizer konverterer ved grænsen til de eksisterende MCL-funktioners cm.
"""

##### TYPING #####

from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Mapping, Optional, Protocol, Sequence

if TYPE_CHECKING:
    from numpy.typing import NDArray

##### LIBARIES #####

from time import sleep, monotonic
from dataclasses import dataclass
from pathlib import Path as FilePath
from importlib import import_module
from types import ModuleType, TracebackType
import sys
import numpy as np
from math import atan2, cos, sin, hypot, radians, degrees, ceil, isfinite, sqrt

Position = tuple[float, float]  # Globalt (x, y), meter
Route = tuple[Position, ...]
Bounds = tuple[float, float, float, float]  # xmin, xmax, ymin, ymax
SensorName = Literal["front", "back", "left", "right"]
PlanStatus = Literal["FOUND", "INVALID_START", "NO_GOAL", "NO_PATH"]
MotionKind = Literal["STOP", "ROTATE", "DRIVE"]


class CameraDevice(Protocol):
    """Den lille del af Picamera2, som bruges efter initialisering."""
    def capture_array(self, name: str) -> NDArray: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...


##### PROGRAM PARAMETERS #####

VERBOSITY : int = 1 # 0 means no debug info, 1 means all, and higher values means less verbosity



##### CONSTANTS #####

from math import pi as PI

ARLO_RADIUS = 0.225 # in meters

WHEEL_2_WHEEL_DIST = 0.381 # center of wheel to center of wheel, not edge to edge

MARKER_SIZE = 0.146 # markørstørrelse i meter på landmarkbox

FOCAL_LENGTH = 1288.9 # Camera focal length from Exercise 3.1

RESOLUTION = (1640, 1232) # x, y in pixels


##### CONFIGURATION AND DATA TYPES #####

@dataclass(frozen=True)
class RallyConfig:
    """Forslagsværdier: banemål, kassestørrelser og tolerancer skal verificeres.

    Figuren i Ex6 viser 4 x 3 m; brugerens tidligere mål var 4 x 4 m.
    Her følger standarden figuren. Margin regnes fra nominelle kassecentre.
    """
    track_width: float = 4.0
    track_depth: float = 3.0
    margin: float = 1.2
    cell_size: float = 0.05
    robot_radius: float = ARLO_RADIUS
    clearance: float = 0.10
    visit_distance: float = 0.40
    goal_gap: float = 0.28
    max_drive_step: float = 0.10
    max_turn_step: float = radians(15)
    heading_tolerance: float = radians(5)
    waypoint_tolerance: float = 0.04
    sonar_stop_distance: float = 0.30  # Fra sensor; skal måles på robotten
    measurement_max_age: float = 0.75
    sonar_ttl: float = 5.0
    max_position_spread: float = 0.15
    max_heading_spread: float = radians(12)
    landmark_max_age: float = 3.0
    max_iter: int = 1500
    goal_sample_rate: int = 10
    confirmations: int = 3

    def __post_init__(self) -> None:
        values = (self.track_width, self.track_depth, self.margin, self.cell_size,
                  self.robot_radius, self.clearance, self.visit_distance, self.goal_gap,
                  self.max_drive_step, self.max_turn_step, self.heading_tolerance,
                  self.waypoint_tolerance, self.sonar_stop_distance,
                  self.measurement_max_age, self.sonar_ttl, self.max_position_spread,
                  self.max_heading_spread, self.landmark_max_age)
        if not all(isfinite(v) and v > 0 for v in values):
            raise ValueError("Konfigurationens afstande/tider/vinkler skal være positive")
        if not self.clearance < self.goal_gap < self.visit_distance:
            raise ValueError("Kræv clearance < goal_gap < visit_distance")
        if self.max_iter < 1 or self.confirmations < 1 or not 0 <= self.goal_sample_rate <= 100:
            raise ValueError("Ugyldige iterations-/bekræftelses-/samplingparametre")

    @property
    def bounds(self) -> Bounds:
        return (0.0, self.track_width + 2*self.margin,
                0.0, self.track_depth + 2*self.margin)

    def landmark_positions(self) -> dict[int, Position]:
        x, y = self.margin, self.margin
        return {1: (x, y), 2: (x, y+self.track_depth),
                3: (x+self.track_width, y),
                4: (x+self.track_width, y+self.track_depth)}


@dataclass(frozen=True)
class PoseEstimate:
    """Robotcentrum i banekoordinater; spredning er ikke en fejlgaranti."""
    x: float
    y: float
    theta: float
    timestamp: float
    position_spread: float
    heading_spread: float
    reliable: bool

    @property
    def position(self) -> Position:
        return (self.x, self.y)


@dataclass(frozen=True)
class ObservationBatch:
    """Ét billede taget ved stilstand; tidspunkt er omtrentligt capture-tidspunkt."""
    landmarks: tuple[Landmark, ...]
    timestamp: float


@dataclass(frozen=True)
class SonarReading:
    sensor: SensorName
    distance: Optional[float]  # Meter fra sensoren; None = ugyldig
    timestamp: float


@dataclass(frozen=True)
class SonarMount:
    """Skal måles: placering fra robotcentrum og retning relativt fremad."""
    forward: float
    left: float
    bearing: float  # Radianer, positiv mod venstre


@dataclass(frozen=True)
class WorldObject:
    id: int
    center: Position
    radius: float  # Fysisk omskrevet cirkel; IKKE inklusive robotradius
    last_seen: float
    uncertainty: float = 0.0
    source: Literal["known", "camera", "sonar"] = "camera"
    expires_at: Optional[float] = None


@dataclass(frozen=True)
class PlanResult:
    status: PlanStatus
    path: Route = ()
    goal: Optional[Position] = None
    reason: str = ""


@dataclass(frozen=True)
class MotionCommand:
    """Kun én kort bevægelse. DRIVE: meter; ROTATE: radianer; STOP: 0."""
    kind: MotionKind
    amount: float = 0.0


@dataclass(frozen=True)
class Decision:
    """Returneres til en fremtidig hardwareløkke; udfører ikke selv motorstyring."""
    state: Literal["LOCALIZE", "SCAN", "MOVE", "BLOCKED", "VISITED", "DONE"]
    command: MotionCommand
    reason: str
    plan: Optional[PlanResult] = None




##### PROGRAM BODY #####

# in case verbosity is false, we want debug calls to not throw an unknown expression error
def debug(*args: object, **kwargs: object) -> None: pass

# if verbosity is true, a debug function which prints to out in a standard way, is defined
if VERBOSITY:
    # defines elapsed_time function for getting time since program start, used by the debug print function
    from time import perf_counter

    _START_TIME = perf_counter()

    def _elapsed_time() -> str:
        elapsed_ms = int((perf_counter() - _START_TIME) * 1000)

        minutes = elapsed_ms // 60_000
        seconds = (elapsed_ms // 1_000) % 60
        milliseconds = elapsed_ms % 1_000

        return f"{minutes:02}:{seconds:02}:{milliseconds:03}"

    # defines the debug print function
    def debug(*values: object, verbosity:int=9) -> None:
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
    def __init__(self, port: str = '/dev/ttyACM0', timeout: float = 0.5) -> None:
        from serial import Serial # type: ignore

        debug("Waiting for serial port connection ...", verbosity=2)

        self.port = port
        
        self.serialRead = Serial(self.port, 9600, timeout=timeout, write_timeout=timeout)

        # Wait if serial port is not open yet
        while not self.serialRead.isOpen():
            sleep(0.25)

        sleep(2)

        debug("Connected to seriel port!", verbosity=2)
                
    def send_command(self, cmd: str, sleep_ms: float = 0.0) -> bytes:
        """Seriel transaktion. Legacy-navnet sleep_ms modtager SEKUNDER."""
        debug(f"send_command({self}, {cmd}, {sleep_ms})", verbosity=1)

        self.serialRead.write(cmd.encode('ascii'))
        sleep(sleep_ms)
        str_val=self.serialRead.readline()
        if not str_val:
            raise TimeoutError("Intet svar fra Arduino")
        return str_val



class Motor:
    def __init__(self, port: Port) -> None:
        self._port = port
        debug("Initialized moters!", verbosity=1)
    
    @staticmethod
    def _power_checker(power:int) -> bool:
        """Checks if a power value is in the set {0, [40;127]}.
           This is an internal utility function."""
        return (power == 0) or (power >=40 and power <=127) 

    def _move(self, powerLeft: int, powerRight: int, dirLeft: int, dirRight: int) -> bytes:
        """Start left motor with motor power powerLeft (in {0, [40;127]} and the numbers must be integer) and direction dirLeft (0=reverse, 1=forward)
           and right motor with motor power powerRight (in {0, [40;127]} and the numbers must be integer) and direction dirRight (0=reverse, 1=forward).
        
           The Arlo robot may blow a fuse if you run the motors at less than 40 in motor power, therefore choose either 
           power = 0 or 40 < power <= 127.
           
           This does NOT use wheel encoders."""
        debug(f"_move({self}, {powerLeft}, {powerRight}, {dirLeft}, {dirRight})", verbosity=1)
        
        # checks for legallity of requisted speed
        if not all(isinstance(p, int) and self._power_checker(p) for p in (powerLeft, powerRight)):
            raise ValueError("Motor power must be integer 0 or 40..127")
        if dirLeft not in (0, 1) or dirRight not in (0, 1):
            raise ValueError("Motor direction must be 0 or 1")

        cmd = 'd' + str(int(powerLeft)) + ',' + str(int(powerRight)) + ',' + str(int(dirLeft)) + ',' + str(int(dirRight)) + '\n'
        return self._port.send_command(cmd)

    def stop(self) -> bytes:
        """Send a stop command to stop motors. Sets the motor power on both wheels to zero.
        
           This does NOT use wheel encoders."""
        debug("stop()", verbosity=3)
        cmd = 's\n'
        return self._port.send_command(cmd)
    
    def move(self, dirLeft: int, dirRight: int) -> bytes:
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

        return self._move(speedLeft, speedRight, 1 if dirLeft==1 else 0, 1 if dirRight==1 else 0)

    def drive(self, meters: float, forward: bool = True, _sleep: bool = True,
              stop_when_done: bool = True) -> float:
        """
        Drives straight forward or backward at ~0.3 meters/second.\\
        Returns the seconds driven.
        """
        debug(f"drive({self}, {meters}, {forward}, {_sleep}, {stop_when_done})", verbosity=3)
        if not isfinite(meters) or meters < 0:
            raise ValueError("Distance must be finite and nonnegative")
        if meters == 0:
            self.stop()
            return 0.0

        meters_pr_sec = 2 * PI * WHEEL_2_WHEEL_DIST / 8   * 3.14/3 # last factor is correction
        
        self.move(2 * forward - 1, 2 * forward - 1)
        t = meters / meters_pr_sec
        if _sleep: sleep(t)
        if stop_when_done: self.stop()
        debug(f"Expected time slept = {t} seconds", verbosity=2)
        return t

    def rotate(self, degrees: float, _sleep: bool = True, stop_when_done: bool = True) -> float:
        """
        Rotates ``degrees`` inplace, at 90°/second, and stops.\\
        Negative rotation is right, positive rotation is left.
        Returns the seconds driven.
        """
        debug(f"rotate({self}, {degrees}, {_sleep}, {stop_when_done})", verbosity=3)
        if not isfinite(degrees):
            raise ValueError("Angle must be finite")

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
            self.stop()
            return 0.0
        t = abs(degrees) / degrees_pr_second / correction_fraction
        if _sleep: sleep(t)
        if stop_when_done: self.stop()
        debug(f"Expected time slept = {t} seconds", verbosity=2)
        return t




class Sonar:
    def __init__(self, port: Port) -> None:
        self._port = port
        debug("Initialized sensors!", verbosity=1)

    def _read_sensor(self, sensorid: int) -> float:
        """Læs Arduinoens millimeter og returnér meter; -1.0 ved ugyldigt svar."""
        debug(f"_read_sensor({self}, {sensorid})", verbosity=1)

        cmd = str(sensorid) + '\n'
        str_val = self._port.send_command(cmd)
        debug(f"Sonar output (str_val) = {str_val}", verbosity=1)
        try:
            value = int(str_val)
        except ValueError:
            return -1.0
        distance = value / 1000.0 if value > 0 else -1.0
        debug(f"Sonar distance in meters = {distance}", verbosity=3)
        return distance
            
    def front(self) -> float:
        """Returnér frontsonarens afstand i meter; -1.0 ved ugyldigt svar."""
        debug(f"front({self})", verbosity=2)
        return self._read_sensor(0)
        
    def back(self) -> float:
        """Returnér bagsonarens afstand i meter; -1.0 ved ugyldigt svar."""
        debug(f"back({self})", verbosity=2)
        return self._read_sensor(1)
        
    def left(self) -> float:
        """Returnér venstre sonars afstand i meter; -1.0 ved ugyldigt svar."""
        debug(f"left({self})", verbosity=2)
        return self._read_sensor(2)
        
    def right(self) -> float:
        """Returnér højre sonars afstand i meter; -1.0 ved ugyldigt svar."""
        debug(f"right({self})", verbosity=2)
        return self._read_sensor(3)

    def read_all(self) -> tuple[SonarReading, ...]:
        """Læs sekventielt ved stilstand. Afstandene er allerede i meter."""
        readings: list[SonarReading] = []
        names: tuple[SensorName, ...] = ("front", "back", "left", "right")
        for sensor_id, name in enumerate(names):
            distance = self._read_sensor(sensor_id)
            readings.append(SonarReading(name, distance if distance > 0 else None, monotonic()))
        return tuple(readings)



class Landmark:
    """Én kasseobservation: center/normal i robotakser, afstand m, vinkel grader.

    Flere sider kan have samme ID. Sammenlæg først efter omregning til kassecentrum.
    Størrelsestabellen er tidligere målte kasser, ikke en garanti for alle eksamenskasser.
    """
    id: int
    center: NDArray[np.float64]
    """Kassecentrum relativt til robotcentrum: 3D-array i meter, x højre/y frem/z op."""
    radius: float
    """Kassens fysiske omskrevne radius i meter, uden robotradius og margin."""
    normal: NDArray[np.float64]
    """Normalvektor for den synlige markørflade i robotakser; dimensionsløs enhedsvektor."""
    distance: float
    """Vandret afstand fra robotcentrum til kassecentrum i meter."""
    angle: float
    """Vinkel fra robottens fremadretning til kassecentrum i grader; positiv mod venstre."""
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

    def __init__(self, id: int, center: NDArray, radius: float, normal: NDArray,
                 distance: float, angle: float) -> None:
        debug(f"Landmark({id}, {center}, {radius}, {normal}, {distance}, {angle})")
        self.id = int(id)
        self.center = np.asarray(center, dtype=float).copy()
        self.radius = float(radius)
        self.normal = np.asarray(normal, dtype=float).copy()
        self.distance = float(distance)
        self.angle = float(angle)
        if (self.center.shape != (3,) or self.normal.shape != (3,)
                or not np.all(np.isfinite(self.center)) or not np.all(np.isfinite(self.normal))
                or not all(isfinite(v) for v in (radius, distance, angle))
                or radius <= 0 or distance < 0):
            raise ValueError("Ugyldig kasseobservation")



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

    def __init__(self) -> None:
        # imports cv2 lib for later use
        global cv2
        import cv2 # type: ignore

        self.cam: Optional[CameraDevice] = self._init_camera()
        debug("Initialized camera!", verbosity=1)

    @staticmethod
    def _init_camera(wait: bool = False) -> CameraDevice:
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

    def close(self) -> None:
        """Stop kameraet og frigiv det én gang, også hvis stop fejler."""
        cam = self.cam
        if cam is None:
            return
        self.cam = None
        try:
            cam.stop()
        finally:
            cam.close()
        debug("Camera closed!", verbosity=1)

    def capture(self) -> NDArray:
        """
        Returnerer billedarray. Kontrollér farveorden på den faktiske kameraopsætning.
        """
        debug("Capturing image!", verbosity=2)
        if self.cam is None:
            raise RuntimeError("Kameraet er lukket")
        return self.cam.capture_array("main") #the capture array function captures next image from the stream
    
    @staticmethod
    def _face_normal(rvec: NDArray) -> NDArray:
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

        tvecs = tvecs @ R.T
        normals = normals @ R.T
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

    def observe(self) -> ObservationBatch:
        """Brug ved stilstand. Tidsstempel før capture er en tilnærmelse, ikke sensor-tid."""
        timestamp = monotonic()
        return ObservationBatch(self.detect_landmarks(), timestamp)



class Robot:
    def __init__(self) -> None:
        self._closed = False
        self._port: Optional[Port] = None
        self.motors: Optional[Motor] = None
        self.camera: Optional[Camera] = None
        debug("Initializing robot ...")
        try:
            self._port = Port()
            self.motors = Motor(self._port)
            self.sonars = Sonar(self._port)
            self.camera = Camera()
        except BaseException:
            try:
                self.close()
            except Exception:
                pass  # Bevar den oprindelige initialiseringsfejl
            raise
        debug("Robot initilized!")

    @property
    def moters(self) -> Optional[Motor]:
        """Kompatibilitetsalias til motors for kode med den tidligere stavemåde."""
        return self.motors

    @property
    def port(self) -> Optional[Port]:
        """Læsbar kompatibilitetsadgang; originalens interne navn er _port."""
        return self._port

    def __enter__(self) -> Robot:
        return self

    def __exit__(self, exc_type: Optional[type[BaseException]],
                 exc_value: Optional[BaseException], traceback: Optional[TracebackType]) -> bool:
        if exc_type is None:
            self.close()
        else:
            try:
                self.close()
            except Exception:
                pass  # En oprydningsfejl må ikke skjule fejlen inde i with-blokken
        return False

    def __del__(self) -> None:
        """Best effort; brug with/finally til rettidig oprydning."""
        try:
            self.close()
        except Exception:
            pass  # Destruktor kan også kaldes efter delvis init eller ved interpreter-exit

    def close(self) -> None:
        """Stop motor, luk kamera, send k og luk port. Gentagne kald er uskadelige."""
        if getattr(self, "_closed", True):
            return
        self._closed = True
        try:
            if self.motors is not None:
                sleep(0.05)
                self.motors.stop()
                sleep(0.1)
        finally:
            try:
                if self.camera is not None:
                    self.camera.close()
            finally:
                if self._port is not None:
                    try:
                        self._port.send_command("k\n")
                    finally:
                        self._port.serialRead.close()
        debug("Robot terminated!", verbosity=1)


##### GEOMETRY #####

def wrap_angle(angle: float) -> float:
    """Korteste vinkel i [-pi, pi)."""
    return (angle + PI) % (2*PI) - PI


def robot_to_world(center: Sequence[float], pose: PoseEstimate) -> Position:
    """Kameraets robotakser (højre, frem) -> globalt (x, y).

    Forward = (cos(theta), sin(theta)); right = (sin(theta), -cos(theta)).
    Kameraets fysiske forskydning er allerede med i Landmark.center.
    """
    right, forward = float(center[0]), float(center[1])
    return (pose.x + forward*cos(pose.theta) + right*sin(pose.theta),
            pose.y + forward*sin(pose.theta) - right*cos(pose.theta))


def distinct_landmarks(observations: Sequence[Landmark]) -> tuple[Landmark, ...]:
    """Én måling pr. kasse/ID pr. billede, så to sider ikke tælles som to landmarks.

    Vælg nærmeste kasseestimat som i den eksisterende MCL-kode. Mere avanceret
    sammenvejning med reprojektionsfejl kan erstatte dette senere.
    """
    selected: dict[int, Landmark] = {}
    for observation in observations:
        if observation.id not in selected or observation.distance < selected[observation.id].distance:
            selected[observation.id] = observation
    return tuple(selected.values())


##### SELF LOCALIZATION #####

class Localizer:
    """Vedvarende MCL-adapter til jeres eksisterende rene hjælpefunktioner.

    Afhænger af ../self-localization/localization_filter.py, motion_model_codex.py
    og Particle. Importerer IKKE selflocalize_s.py eller robot.py.
    Der er ingen reset ved nyt mål eller ny RRT-rute.

    Ind/ud: m, rad. Kun backend: cm, rad. Observationer refererer her til
    KASSECENTRUM ved ROBOTCENTRUM, så backendens kamera-offset sættes til nul.
    """
    config: RallyConfig
    backend: ModuleType
    particles: list[object]

    def __init__(self, config: RallyConfig, num_particles: int = 1000) -> None:
        self.config = config
        directory = FilePath(__file__).resolve().parents[1] / "self-localization"
        if not (directory / "localization_filter.py").is_file():
            raise FileNotFoundError(f"MCL-hjælpefiler mangler i {directory}")
        # Legacy-modulerne bruger absolutte søskendeimports; behold deres mappe
        # på søgestien. Alle de indlæste filterfiler er uden hardwareopstart.
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
        self.backend = import_module("localization_filter")
        if FilePath(self.backend.__file__).resolve() != directory / "localization_filter.py":
            raise ImportError("Et andet localization_filter-modul er allerede indlæst")
        self.num_particles = num_particles
        self.known_cm = {i: (100*x, 100*y) for i, (x, y) in config.landmark_positions().items()}
        self.reset()

    def reset(self) -> None:
        """Kun ved opstart/tabt lokalisering. Rally.reset_localization rydder også objektkort."""
        self.particles = self.backend.initialize_particles(
            self.num_particles, tuple(100*v for v in self.config.bounds))
        self.timestamp = float("-inf")
        self.last_landmark_time = float("-inf")
        self.seen_ids: set[int] = set()
        self.seeded = False
        self.consistent = False

    def predict(self, velocity: float, angular_velocity: float,
                duration: float, timestamp: float) -> None:
        """Udført/estimeret kontrol: m/s, rad/s, sekunder, bevægelsens sluttid.

        Kald præcis én gang pr. bevægelsesinterval, før næste kamerakorrektion.
        Kalibreret hastighed er et estimat; det er ikke encoder-odometri.
        """
        if (not all(isfinite(v) for v in (velocity, angular_velocity, duration, timestamp))
                or duration < 0 or timestamp < self.timestamp):
            raise ValueError("Ugyldig bevægelse eller tidsrækkefølge")
        self.particles = self.backend.mcl_step(
            self.particles, (100*velocity, angular_velocity), None, duration, self.known_cm)
        self.timestamp = timestamp

    def update(self, batch: ObservationBatch) -> PoseEstimate:
        """Brug alle kendte ID'er, uanset aktuel fase. Afvis genbrug af samme billede."""
        if not isfinite(batch.timestamp) or batch.timestamp <= self.timestamp:
            raise ValueError("Billedet skal være nyere end seneste filteropdatering")
        observations = [o for o in distinct_landmarks(batch.landmarks) if o.id in self.known_cm]
        measurements = ([o.id for o in observations],
                        [100*hypot(float(o.center[0]), float(o.center[1])) for o in observations],
                        [atan2(-float(o.center[0]), float(o.center[1])) for o in observations])
        proposal = None
        if not self.seeded and len(observations) >= 2:
            proposal = self.backend.initialize_from_observations(
                self.num_particles, measurements, self.known_cm,
                sigma_distance=10.0, sigma_angle=0.10)
        if proposal is not None:
            self.particles = proposal
            self.seeded = True  # Samme billede må ikke vægte startforslaget igen
        else:
            self.particles = self.backend.mcl_step(
                self.particles, (0.0, 0.0), measurements, 0.0, self.known_cm,
                sigma_distance=10.0, sigma_angle=0.10)
        self.timestamp = batch.timestamp
        if observations:
            self.seen_ids.update(o.id for o in observations)
            self.last_landmark_time = batch.timestamp
            estimate = self.backend.estimate_pose(self.particles)
            # Et snævert, men åbenlyst forkert estimat skal ikke skrives på kortet.
            self.consistent = all(
                abs(hypot(self.known_cm[o.id][0]-estimate.getX(),
                          self.known_cm[o.id][1]-estimate.getY())/100-o.distance) < 0.35
                and abs(wrap_angle(atan2(self.known_cm[o.id][1]-estimate.getY(),
                                        self.known_cm[o.id][0]-estimate.getX())
                                   - estimate.getTheta()-radians(o.angle))) < 0.30
                for o in observations)
        return self.estimate(batch.timestamp)

    def estimate(self, now: float) -> PoseEstimate:
        pose = self.backend.estimate_pose(self.particles)
        spread_cm, heading_spread = self.backend.pose_spread(self.particles)
        x, y = pose.getX()/100, pose.getY()/100
        xmin, xmax, ymin, ymax = self.config.bounds
        reliable = (self.consistent and len(self.seen_ids) >= 2
                    and 0 <= now-self.last_landmark_time <= self.config.landmark_max_age
                    and spread_cm/100 <= self.config.max_position_spread
                    and heading_spread <= self.config.max_heading_spread
                    and xmin <= x < xmax and ymin <= y < ymax)
        return PoseEstimate(x, y, wrap_angle(pose.getTheta()), self.timestamp,
                            spread_cm/100, heading_spread, reliable)


##### WORLD MAP #####

class WorldMap:
    """Vedvarende objekter. RRT må nulstilles uden at slette dette kort.

    Kendte landmarks står fast. Kameraobjekter opdateres efter ID.
    Sonar giver midlertidige, usikre objekter; ingen automatisk vægklassifikation.
    """
    objects: dict[int, WorldObject]

    def __init__(self, config: RallyConfig) -> None:
        self.config = config
        self.objects = {
            i: WorldObject(i, p, Landmark.MAX_BOX_SIZES[i]/sqrt(2), 0.0, source="known")
            for i, p in config.landmark_positions().items()}
        self.version = 0
        self.next_sonar_id = -1

    def clear_observed(self) -> None:
        """Efter reset/stor posekorrektion: behold faste ankre, fjern usikre placeringer."""
        self.objects = {i: o for i, o in self.objects.items() if o.source == "known"}
        self.version += 1

    def update(self, batch: ObservationBatch, pose: PoseEstimate) -> None:
        if not pose.reliable or abs(pose.timestamp-batch.timestamp) > self.config.measurement_max_age:
            return  # En usikker pose må ikke blive et permanent kortpunkt
        for observation in distinct_landmarks(batch.landmarks):
            if observation.id <= 4:
                continue
            center = robot_to_world(observation.center, pose)
            uncertainty = pose.position_spread + observation.distance*abs(sin(pose.heading_spread)) + 0.05
            self.objects[observation.id] = WorldObject(
                observation.id, center, observation.radius, batch.timestamp, uncertainty)
            self.version += 1

    def add_sonar(self, reading: SonarReading, mount: SonarMount,
                  pose: PoseEstimate, now: float) -> None:
        """Groft ekkoområde, ikke præcis væg. Sensorens montage skal angives målt.

        Kald ved stilstand og kun for relevante, gyldige nærmålinger.
        Udskifter ikke kendte kasser med en ny sonar-identitet.
        """
        if (not pose.reliable or reading.distance is None or reading.distance <= 0
                or not isfinite(reading.distance)
                or not 0 <= now-reading.timestamp <= self.config.measurement_max_age
                or abs(pose.timestamp-reading.timestamp) > self.config.measurement_max_age):
            return
        forward = mount.forward + reading.distance*cos(mount.bearing)
        left = mount.left + reading.distance*sin(mount.bearing)
        center = robot_to_world((-left, forward), pose)
        # Forslagsværdi for et bredt ekkoområde, ikke en målt sensoråbningsvinkel.
        radius = max(0.08, reading.distance*sin(radians(15)))
        uncertainty = pose.position_spread + reading.distance*abs(sin(pose.heading_spread))
        key = next((i for i, o in self.objects.items() if o.source == "sonar"
                    and hypot(o.center[0]-center[0], o.center[1]-center[1]) < 0.20),
                   self.next_sonar_id)
        if key == self.next_sonar_id:
            self.next_sonar_id -= 1
        self.objects[key] = WorldObject(key, center, radius, reading.timestamp,
                                        uncertainty, "sonar", now+self.config.sonar_ttl)
        self.version += 1

    def active_objects(self, now: float) -> tuple[WorldObject, ...]:
        expired = [i for i, o in self.objects.items() if o.expires_at is not None and o.expires_at <= now]
        for i in expired:
            del self.objects[i]
        if expired:
            self.version += 1
        return tuple(self.objects.values())


##### GRID - ADAPTED FROM EXERCISE 4.2 #####

class Grid:
    """Bool-grid: True = blokeret; False = ingen kendt blokering, IKKE bevist frit.

    Matrixindeks er [x, y], som i Exercise_4_2_grid.py. Rasterisering tester
    hele cellen mod objekter. Banekanten indrykkes med robotradius + margin.
    """
    grid_matrix: NDArray[np.bool_]

    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float,
                 grid_cell_size: float) -> None:
        if (not all(isfinite(v) for v in (x_min, x_max, y_min, y_max, grid_cell_size))
                or x_min >= x_max or y_min >= y_max or grid_cell_size <= 0):
            raise ValueError("Ugyldige gridgrænser/cellestørrelse")
        self.x_limits = (x_min, x_max)
        self.y_limits = (y_min, y_max)
        self.grid_cell_size = grid_cell_size
        self.low = np.array([x_min, y_min])
        self.high = np.array([x_max, y_max])
        shape = np.ceil(np.nextafter((self.high-self.low)/grid_cell_size, -np.inf)).astype(int)
        self.grid_matrix = np.zeros(tuple(shape), dtype=bool)
        self.number_of_celles = list(map(int, shape))
        indices = np.moveaxis(np.indices(tuple(shape)), 0, -1)
        self.cell_min = self.low + indices*grid_cell_size
        self.cell_max = np.minimum(self.cell_min+grid_cell_size, self.high)
        self.departure: Optional[tuple[Position, float]] = None
        self._occupied_min: Optional[NDArray] = None
        self._occupied_max: Optional[NDArray] = None

    def block_circle(self, center: Position, radius: float) -> None:
        nearest = np.clip(np.asarray(center), self.cell_min, self.cell_max)
        self.grid_matrix |= np.sum((nearest-center)**2, axis=-1) <= radius**2
        self._occupied_min = None

    def obstacles(self, objects: Sequence[WorldObject], robot_radius: float,
                  buffer: float = 0.10) -> None:
        """Udvid hver fysisk forhindring præcis én gang med robot + margin + usikkerhed."""
        for obj in objects:
            self.block_circle(obj.center, obj.radius+robot_radius+buffer+obj.uncertainty)

    def block_boundary(self, padding: float) -> None:
        self.grid_matrix |= np.any(self.cell_min < self.low+padding, axis=-1)
        self.grid_matrix |= np.any(self.cell_max > self.high-padding, axis=-1)
        self._occupied_min = None

    def contains(self, point: Position) -> bool:
        p = np.asarray(point)
        return bool(p.shape == (2,) and np.all(np.isfinite(p))
                    and np.all(p >= self.low) and np.all(p < self.high))

    def segment_is_free(self, start: Position, end: Position) -> bool:
        """Test ALLE berørte optagede celler, også slutpunkt/hjørner og nul længde.

        Rettelsen bygger på Exercise_4_2_rrt_Chat1.py. Ingen klipning af
        punkter uden for kortet. Kaldes også efter ruteforenkling.
        """
        if not self.contains(start) or not self.contains(end):
            return False
        a, b = np.asarray(start), np.asarray(end)
        direction = b-a
        if self.departure is not None:
            center, radius = self.departure
            relative = a-center
            if np.linalg.norm(relative) < radius:
                # Udgang fra netop besøgt landmark skal gå væk fra centrum.
                if np.dot(relative, direction) < -1e-12:
                    return False
            else:
                length2 = float(direction @ direction)
                t = float(np.clip(-np.dot(relative, direction)/length2, 0, 1)) if length2 else 0.0
                if np.linalg.norm(relative+t*direction) <= radius:
                    return False  # Ingen genindtræden efter at have forladt området
        if self._occupied_min is None:
            occupied = self.grid_matrix
            self._occupied_min = self.cell_min[occupied]-1e-12
            self._occupied_max = self.cell_max[occupied]+1e-12
        low, high = self._occupied_min, self._occupied_max
        enter, leave = np.zeros(len(low)), np.ones(len(low))
        possible = np.ones(len(low), dtype=bool)
        for axis in range(2):
            if direction[axis] == 0:
                possible &= (a[axis] >= low[:, axis]) & (a[axis] <= high[:, axis])
            else:
                t1 = (low[:, axis]-a[axis])/direction[axis]
                t2 = (high[:, axis]-a[axis])/direction[axis]
                enter = np.maximum(enter, np.minimum(t1, t2))
                leave = np.minimum(leave, np.maximum(t1, t2))
        return not bool(np.any(possible & (enter <= leave)))

    @classmethod
    def from_world(cls, world: WorldMap, target_id: int, departing_id: Optional[int],
                   now: float) -> Grid:
        c = world.config
        grid = cls(*c.bounds, c.cell_size)
        objects = world.active_objects(now)
        grid.obstacles(objects, c.robot_radius, c.clearance)
        grid.block_boundary(c.robot_radius+c.clearance)
        for obj in objects:
            if obj.source != "known" or obj.id == target_id:
                continue
            radius = obj.radius+c.robot_radius+c.visit_distance+c.clearance
            if obj.id == departing_id:
                grid.departure = (obj.center, radius)
            else:
                grid.block_circle(obj.center, radius)
        return grid


##### RRT - ADAPTED FROM EXERCISE_4_2_RRT.PY #####

class Node:
    def __init__(self, position: Position, parent: Optional[Node] = None) -> None:
        self.position = (float(position[0]), float(position[1]))
        self.parent = parent

    def eucl_dist(self, target_node: Node) -> float:
        return hypot(target_node.position[0]-self.position[0], target_node.position[1]-self.position[1])


class RRT:
    """Kun planlægning. Ingen motorer, kamera, JSON-læsning eller globale arlo-objekter."""
    def __init__(self, start: Position, goal: Position, map: Grid,
                 moving_dist_limit: float = 0.5, goal_sample_rate: int = 10,
                 max_iter: int = 1500, rng: Optional[np.random.Generator] = None) -> None:
        if moving_dist_limit <= 0 or not isfinite(moving_dist_limit) or max_iter < 1:
            raise ValueError("Ugyldige RRT-parametre")
        if not 0 <= goal_sample_rate <= 100:
            raise ValueError("goal_sample_rate skal være 0..100")
        self.start, self.end = Node(start), Node(goal)
        self.map = map
        self.moving_dist_limit = moving_dist_limit
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.rng = rng if rng is not None else np.random.default_rng()
        self.node_list: list[Node] = []

    def random_node(self) -> Node:
        if self.rng.random() < self.goal_sample_rate/100:
            return Node(self.end.position)
        sample = self.rng.uniform(self.map.low, self.map.high)
        return Node((float(sample[0]), float(sample[1])))

    def nearest_node(self, node_list: Sequence[Node], random_node: Node) -> Node:
        return min(node_list, key=lambda node: node.eucl_dist(random_node))

    def steering(self, old_node: Node, target: Node, moving_dist_limit: float) -> Node:
        distance = old_node.eucl_dist(target)
        fraction = min(1.0, moving_dist_limit/distance) if distance else 0.0
        a, b = old_node.position, target.position
        return Node((a[0]+fraction*(b[0]-a[0]), a[1]+fraction*(b[1]-a[1])), old_node)

    def check_collision(self, old_node: Node, new_node: Node) -> bool:
        """Som i øvelsen: True betyder FRI passage. Bruger geometrien, ikke node.distance."""
        return self.map.segment_is_free(old_node.position, new_node.position)

    def generate_final_course(self, goal_node: Node) -> Route:
        path: list[Position] = []
        node: Optional[Node] = goal_node
        while node is not None:
            path.append(node.position)
            node = node.parent
        return tuple(reversed(path))

    def path_planning(self) -> Optional[Route]:
        self.node_list = [self.start]
        if not self.check_collision(self.start, self.start) or not self.check_collision(self.end, self.end):
            return None
        if self.check_collision(self.start, self.end):
            return (self.start.position,) if self.start.position == self.end.position else (self.start.position, self.end.position)
        for _ in range(self.max_iter):
            sample = self.random_node()
            nearest = self.nearest_node(self.node_list, sample)
            new = self.steering(nearest, sample, self.moving_dist_limit)
            if new.eucl_dist(nearest) < 1e-10 or not self.check_collision(nearest, new):
                continue
            self.node_list.append(new)
            if new.eucl_dist(self.end) <= self.moving_dist_limit and self.check_collision(new, self.end):
                return self.generate_final_course(Node(self.end.position, new))
        return None

    def simpler_path(self, path: Sequence[Position]) -> Route:
        if not path:
            return ()
        if any(not self.map.segment_is_free(a, b) for a, b in zip(path, path[1:])):
            raise ValueError("Kan ikke forenkle en allerede blokeret rute")
        result = [path[0]]
        index = 0
        while index < len(path)-1:
            following = next(j for j in range(len(path)-1, index, -1)
                             if self.map.segment_is_free(path[index], path[j]))
            result.append(path[following])
            index = following
        return tuple(result)


##### MISSION - L1, L2, L3, L4, L1 #####

class RallyMission:
    """Målet ændres kun ved friske, gentagne observationer tæt nok på kassen."""
    sequence: tuple[int, ...] = (1, 2, 3, 4, 1)

    def __init__(self, config: RallyConfig) -> None:
        self.config = config
        self.phase_index = 0
        self.departing_id: Optional[int] = None
        self.confirmation_count = 0
        self.last_confirmation_time = float("-inf")
        self.visit_history: list[tuple[int, float]] = []

    @property
    def done(self) -> bool:
        return self.phase_index == len(self.sequence)

    @property
    def target_id(self) -> Optional[int]:
        return None if self.done else self.sequence[self.phase_index]

    def confirm_visit(self, batch: ObservationBatch, now: float) -> bool:
        """Forsigtig afstandstest med kvadratisk kassemodel og 5 cm målereserve.

        Omskrevet cirkel giver en nedre afstandsgrænse (kollisionskontrol).
        Indskrevet cirkel giver en øvre grænse (kan bekræfte besøg).
        Kalibrér målereserven; det er ikke en statistisk fejlgaranti.
        """
        if self.done or batch.timestamp <= self.last_confirmation_time:
            return False
        self.last_confirmation_time = batch.timestamp
        observation = next((o for o in distinct_landmarks(batch.landmarks)
                            if o.id == self.target_id), None)
        if observation is None or not 0 <= now-batch.timestamp <= self.config.measurement_max_age:
            self.confirmation_count = 0
            return False
        distance = hypot(float(observation.center[0]), float(observation.center[1]))
        half_side = Landmark.MAX_BOX_SIZES[observation.id]/2
        gap_lower = distance-self.config.robot_radius-observation.radius
        gap_upper = distance-self.config.robot_radius-half_side
        if gap_lower <= 0.05 or gap_upper+0.05 > self.config.visit_distance:
            self.confirmation_count = 0
            return False
        self.confirmation_count += 1
        if self.confirmation_count < self.config.confirmations:
            return False
        target = self.target_id
        assert target is not None
        self.visit_history.append((target, batch.timestamp))
        self.departing_id = target
        self.phase_index += 1
        self.confirmation_count = 0
        return True


##### NAVIGATION - SPLIT FROM EXERCISE_4_2.ROBOT_PATH #####

def sonar_blocks(readings: Sequence[SonarReading], now: float, config: RallyConfig,
                 sensors: Sequence[SensorName]) -> bool:
    """Ukendt, manglende, gammel eller for kort afstand kræver stop."""
    latest = {reading.sensor: reading for reading in readings}
    for name in sensors:
        reading = latest.get(name)
        if (reading is None or reading.distance is None or not isfinite(reading.distance)
                or reading.distance <= config.sonar_stop_distance
                or not 0 <= now-reading.timestamp <= config.measurement_max_age):
            return True
    return False


class Scanner:
    """Drej-stop-mål i en omdrejning; caller tager et nyt billede mellem kommandoer"""
    def __init__(self, config: RallyConfig) -> None:
        self.config = config
        self.rotated = 0.0

    def restart(self) -> None:
        self.rotated = 0.0

    def next_command(self, readings: Sequence[SonarReading], now: float) -> MotionCommand:
        if (self.rotated >= 2*PI-1e-9
                or sonar_blocks(readings, now, self.config, ("front", "back", "left", "right"))):
            return MotionCommand("STOP")
        return MotionCommand("ROTATE", min(self.config.max_turn_step, 2*PI-self.rotated))

    def record_rotation(self, radians_turned: float) -> None:
        """Kald efter udført scanningstrin, aldrig blot fordi kommandoen blev foreslået."""
        self.rotated += abs(radians_turned)


class Navigator:
    """RRT vælger vejen; next_command bruger AKTUEL pose og højst et kort stræk.

    robot_path() fra Exercise 4 kørte hele ruten med antaget startretning nul.
    Her adskilles planlægning, bevægelsesvalg og fysisk udførelse.
    """
    def __init__(self, config: RallyConfig, rng: Optional[np.random.Generator] = None) -> None:
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()

    def goal_candidates(self, target: WorldObject, pose: PoseEstimate, grid: Grid) -> tuple[Position, ...]:
        """Prøv flere sider af kassen; mål er robotcentrum, aldrig kassecentrum."""
        half_side = Landmark.MAX_BOX_SIZES[target.id]/2
        radius = self.config.robot_radius+half_side+self.config.goal_gap
        angle = atan2(pose.y-target.center[1], pose.x-target.center[0])
        candidates = [(target.center[0]+radius*cos(a), target.center[1]+radius*sin(a))
                      for a in np.linspace(angle, angle+2*PI, 16, endpoint=False)]
        candidates = [p for p in candidates if grid.segment_is_free(p, p)]
        return tuple(sorted(candidates, key=lambda p: hypot(p[0]-pose.x, p[1]-pose.y)))

    def plan(self, pose: PoseEstimate, world: WorldMap, mission: RallyMission, now: float) -> PlanResult:
        if not pose.reliable or not 0 <= now-pose.timestamp <= self.config.measurement_max_age:
            return PlanResult("INVALID_START", reason="Robotpositionen er usikker eller gammel")
        if mission.target_id is None:
            return PlanResult("NO_GOAL", reason="Løbet er færdigt")
        grid = Grid.from_world(world, mission.target_id, mission.departing_id, now)
        if not grid.segment_is_free(pose.position, pose.position):
            return PlanResult("INVALID_START", reason="Start er blokeret; mål igen, fjern ikke blot blokeringen")
        candidates = self.goal_candidates(world.objects[mission.target_id], pose, grid)
        if not candidates:
            return PlanResult("NO_GOAL", reason="Ingen fri ankomstposition omkring målkassen")
        # max_iter er samlet budget for kandidatforsøg, ikke 16 fulde RRT-kørsler.
        budget = max(1, self.config.max_iter//len(candidates))
        for goal in candidates:
            planner = RRT(pose.position, goal, grid, goal_sample_rate=self.config.goal_sample_rate,
                          max_iter=budget, rng=self.rng)
            path = planner.path_planning()
            if path is not None:
                return PlanResult("FOUND", planner.simpler_path(path), goal)
        return PlanResult("NO_PATH", reason="RRT fandt ingen rute inden for forsøgsbudgettet")

    def next_command(self, pose: PoseEstimate, plan: PlanResult, grid: Grid,
                     readings: Sequence[SonarReading], now: float) -> MotionCommand:
        if (plan.status != "FOUND" or not pose.reliable
                or not 0 <= now-pose.timestamp <= self.config.measurement_max_age):
            return MotionCommand("STOP")
        target = next((p for p in plan.path
                       if hypot(p[0]-pose.x, p[1]-pose.y) > self.config.waypoint_tolerance), None)
        if target is None or not grid.segment_is_free(pose.position, target):
            return MotionCommand("STOP")
        dx, dy = target[0]-pose.x, target[1]-pose.y
        error = wrap_angle(atan2(dy, dx)-pose.theta)
        if abs(error) > self.config.heading_tolerance:
            if sonar_blocks(readings, now, self.config, ("front", "back", "left", "right")):
                return MotionCommand("STOP")
            return MotionCommand("ROTATE", float(np.clip(error, -self.config.max_turn_step, self.config.max_turn_step)))
        if sonar_blocks(readings, now, self.config, ("front", "left", "right")):
            return MotionCommand("STOP")
        return MotionCommand("DRIVE", min(hypot(dx, dy), self.config.max_drive_step))


##### RALLY COORDINATOR #####

class Rally:
    """Samler beslutninger, men ejer ingen hardware og starter ikke motorerne.

    Hardwareløkken skal: stop -> capture -> sonar -> decide -> udfør ét trin
    -> predict med udført kontrol og faktisk tidsinterval -> gentag.
    Friske målinger/pose skal dække samme stilstandsperiode. Planlægning kan
    tage tid: kontrollér sensorernes alder igen før fysisk udførelse.
    """
    def __init__(self, config: Optional[RallyConfig] = None) -> None:
        self.config = config if config is not None else RallyConfig()
        self.localizer = Localizer(self.config)
        self.world = WorldMap(self.config)
        self.mission = RallyMission(self.config)
        self.navigator = Navigator(self.config)
        self.scanner = Scanner(self.config)

    def reset_localization(self) -> None:
        """Recovery; bevarer løbets fase, men forkaster kortdata med usikker placering."""
        self.localizer.reset()
        self.world.clear_observed()
        self.scanner.restart()

    def observe(self, batch: ObservationBatch) -> PoseEstimate:
        # Sammenlign med prediction ved samme tidspunkt, ikke med positionen før kørsel.
        predicted = self.localizer.estimate(batch.timestamp)
        pose = self.localizer.update(batch)
        if predicted.reliable and pose.reliable:
            jump = hypot(pose.x-predicted.x, pose.y-predicted.y)
            turn = abs(wrap_angle(pose.theta-predicted.theta))
            if jump > 0.25 or turn > radians(15):
                self.world.clear_observed()
        elif not pose.reliable:
            # Konservativ prototype: behold ikke gammel kamerageometri gennem lokaliseringstab.
            self.world.clear_observed()
        self.world.update(batch, pose)
        return pose

    def decide(self, batch: ObservationBatch, readings: Sequence[SonarReading], now: float,
               sonar_mounts: Optional[Mapping[SensorName, SonarMount]] = None) -> Decision:
        stop = MotionCommand("STOP")
        if self.mission.done:
            return Decision("DONE", stop, "L1, L2, L3, L4, L1 er besøgt")
        if not 0 <= now-batch.timestamp <= self.config.measurement_max_age:
            return Decision("BLOCKED", stop, "Tag et nyt billede")
        pose = self.observe(batch)
        if self.mission.confirm_visit(batch, now):
            return Decision("DONE" if self.mission.done else "VISITED", stop,
                            "Besøg bekræftet; fase ændret, filter og kort beholdes")
        if not pose.reliable:
            command = self.scanner.next_command(readings, now)
            return Decision("LOCALIZE", command,
                            "Scan i små trin; ved STOP undersøges sonar eller afsluttet scanning")
        self.scanner.restart()
        # Sonarstop opdaterer kortet, hvis målte sensorpositioner er til rådighed.
        for reading in readings:
            if (reading.distance is not None and reading.distance <= self.config.sonar_stop_distance
                    and sonar_mounts is not None and reading.sensor in sonar_mounts):
                self.world.add_sonar(reading, sonar_mounts[reading.sensor], pose, now)
        # Enkel første version: nyt RRT-træ pr. beslutning, samme vedvarende kort/filter.
        plan = self.navigator.plan(pose, self.world, self.mission, now)
        if plan.status != "FOUND":
            return Decision("BLOCKED", stop, plan.reason, plan)
        assert self.mission.target_id is not None
        grid = Grid.from_world(self.world, self.mission.target_id, self.mission.departing_id, now)
        command = self.navigator.next_command(pose, plan, grid, readings, now)
        if command.kind == "STOP":
            # Målet kan være nået geometrisk uden synlig markør. Ret kameraet mod kassen.
            if plan.goal is not None and hypot(pose.x-plan.goal[0], pose.y-plan.goal[1]) <= self.config.waypoint_tolerance:
                center = self.world.objects[self.mission.target_id].center
                error = wrap_angle(atan2(center[1]-pose.y, center[0]-pose.x)-pose.theta)
                if (abs(error) > self.config.heading_tolerance
                        and not sonar_blocks(readings, now, self.config, ("front", "back", "left", "right"))):
                    command = MotionCommand("ROTATE", float(np.clip(error, -self.config.max_turn_step, self.config.max_turn_step)))
                return Decision("SCAN", command, "Bekræft målets ID og afstand i nye billeder", plan)
            return Decision("BLOCKED", stop, "Sonar, ny blokering eller manglende fremdrift; mål igen", plan)
        return Decision("MOVE", command, "Udfør ét kort trin, stop og mål igen", plan)


##### MAIN - OFFLINE EXAMPLE #####

def main() -> None:
    """Konkret eksempel på WorldMap -> Grid -> RRT -> MotionCommand uden hardware.

    Ingen Robot() oprettes. Fuld fysisk løkke og bevægelsesrapportering er
    bevidst næste integrationsopgave; funktionen her er ikke en robotkørsel.
    """
    config = RallyConfig()
    world = WorldMap(config)
    mission = RallyMission(config)
    # Kun demo: antag at første L1 er besøgt, og robotten står nedenfor L1.
    mission.phase_index = 1
    mission.departing_id = 1
    now = 100.0
    pose = PoseEstimate(1.2, 0.55, PI/2, now, 0.02, 0.02, True)
    world.objects[7] = WorldObject(7, (2.0, 2.5), 0.25, now)
    navigator = Navigator(config, np.random.default_rng(7))
    plan = navigator.plan(pose, world, mission, now)
    grid = Grid.from_world(world, 2, 1, now)
    readings = tuple(SonarReading(name, 2.0, now) for name in ("front", "back", "left", "right"))
    command = navigator.next_command(pose, plan, grid, readings, now)
    print("rally_T: offline designforslag, ingen hardware startet")
    print(f"Mål: L{mission.target_id}; grid: {grid.grid_matrix.shape}; plan: {plan.status}")
    print(f"Rute: {plan.path}")
    print(f"Næste forslag: {command}")


if __name__ == "__main__":
    main()
