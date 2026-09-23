# Arlo Robot Controller

from time import sleep
import serial # type: ignore
from math import gcd
import cv2 # type: ignore
import picamera2 # type: ignore
import numpy as np


### CONSTANTS
FOCAL_LENGTH = 1288.9 # Camera focal length from Exercise 3.1
CX, CY = 1640/2, 1232/2 # Camera center in pixels
CAMERA_MATRIX = np.array([
    [FOCAL_LENGTH, 0, CX], 
    [0, FOCAL_LENGTH, CY], 
    [0, 0, 1]
], dtype=np.float32) # 3x3 matrix 
DISTORTION_MATRIX = np.zeros((5, 1), dtype=np.float32) # Zero vector ; assumes no camera (lens) distortion

MARKER_SIZE = 0.146 # markørstørrelse i meter på landmarkbox

ARLO_RADIUS = 0.225 # in meters


class Robot(object):
    """Defines the Arlo robot API
    
       DISCLAIMER: This code does not contain error checking - it is the responsibility
       of the caller to ensure proper parameters and not to send commands to the 
       Arduino too frequently (give it time to process the command by adding a short sleep wait
       statement). Failure to do some may lead to strange robot behaviour.
       
       In elif userinput == you experience trouble - consider using only commands that do not use the wheel 
       encoders.
    
       This class is not thread-safe and you should only use it from one thread. However, this does not have
       to be the main thread, and you can move your robot communication code into a separate thread.
    """ 
    def __init__(self, port = '/dev/ttyACM0'):
        """The constructor port parameter can be changed from default value if you want
           to control the robot directly from your labtop (instead of from the on-board raspberry 
           pi). The value of port should point to the USB port on which the robot Arduino is connected."""
        self.port = port
        
        #self.serialRead = serial.Serial(self.port,9600, timeout=1) # 1 sec. timeout, wait until data is received or until timeout
        self.serialRead = serial.Serial(self.port,9600, timeout=None) # No timeout, wait forever or until data is received

        # Wait if serial port is not open yet
        while not self.serialRead.isOpen():
            sleep(1)

        print("Waiting for serial port connection ...")
        sleep(2)

        print("Running ...")
        
    def __del__(self):
        print("Shutting down the robot ...")
        
        sleep(0.05)
        print(self.stop())
        sleep(0.1)
                
        cmd='k\n'
        print((self.send_command(cmd)))
        self.serialRead.close()
        
        
    def send_command(self, cmd, sleep_ms=0.0):
        """Sends a command to the Arduino robot controller"""
        self.serialRead.write(cmd.encode('ascii'))
        sleep(sleep_ms)
        str_val=self.serialRead.readline()
        return str_val


    @staticmethod
    def _power_checker(power):
        """Checks if a power value is in the set {0, [40;127]}.
           This is an internal utility function."""
        return  (power == 0) or (power >=40 and power <=127) 

        
    def go_diff(self, powerLeft, powerRight, dirLeft, dirRight):
        """Start left motor with motor power powerLeft (in {0, [40;127]} and the numbers must be integer) and direction dirLeft (0=reverse, 1=forward)
           and right motor with motor power powerRight (in {0, [40;127]} and the numbers must be integer) and direction dirRight (0=reverse, 1=forward).
        
           The Arlo robot may blow a fuse if you run the motors at less than 40 in motor power, therefore choose either 
           power = 0 or 40 < power <= 127.
           
           This does NOT use wheel encoders."""
        
        if (not self._power_checker(powerLeft)) or (not self._power_checker(powerRight)):
            print("WARNING: Read the docstring of Robot.go_diff()!")
            return ""
        else:
            cmd = 'd' + str(int(powerLeft)) + ',' + str(int(powerRight)) + ',' + str(int(dirLeft)) + ',' + str(int(dirRight)) + '\n'
            return self.send_command(cmd)


    def stop(self):
        """Send a stop command to stop motors. Sets the motor power on both wheels to zero.
        
           This does NOT use wheel encoders."""
        cmd='s\n'
        return self.send_command(cmd)


    
    def read_sensor(self, sensorid):
        """Send a read sensor command with sensorid and return sensor value. 
           Will return -1, if error occurs."""
        cmd=str(sensorid) + '\n'
        str_val=self.send_command(cmd)
        if len(str_val) > 0:
            return int(str_val)
        else:
            return -1
            
    def read_front_ping_sensor(self):
        """Read the front sonar ping sensor and return the measured range in milimeters [mm]"""
        return self.read_sensor(0)
        
    def read_back_ping_sensor(self):
        """Read the back sonar ping sensor and return the measured range in milimeters [mm]"""
        return self.read_sensor(1)
        
    def read_left_ping_sensor(self):
        """Read the left sonar ping sensor and return the measured range in milimeters [mm]"""
        return self.read_sensor(2)
        
    def read_right_ping_sensor(self):
        """Read the right sonar ping sensor and return the measured range in milimeters [mm]"""
        return self.read_sensor(3)

    
    def read_left_wheel_encoder(self):
        """Reads the left wheel encoder counts since last reset_encoder_counts command.
           The encoder has 144 counts for one complete wheel revolution."""
        cmd='e0\n'
        return self.send_command(cmd, 0.045)

    def read_right_wheel_encoder(self):
        """Reads the right wheel encoder counts since last clear reset_encoder_counts command.
           The encoder has 144 counts for one complete wheel revolution."""
        cmd='e1\n'
        return self.send_command(cmd, 0.045)

    def reset_encoder_counts(self):
        """Reset the wheel encoder counts."""
        cmd='c\n'
        return self.send_command(cmd)
    


    ### CAMERA

    def start_camera(self):
        self.cam = picamera2.Picamera2() #open camera
        config = self.cam.create_video_configuration({ # define camera config suitable for recording video
            "size": (1640, 1232), 
            "format": "RGB888"
        }) 

        self.cam.configure(config) #use config
        self.cam.start(show_preview=False) #start camera (turn on)
        sleep(1) #wait for camera to start

    def take_picture(self):
        """Takes image in RBG format and return array of shape (hight, width, rbg)
        Returns (ids, rvecs, tvecs)"""
        return self.cam.capture_array("main") #the capture array function captures next image from the stream
    
    def picDetectMarkersPose(self):
        """Takes picture, detects markers in image and estimates poses for detected markers."""
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

        frame = self.take_picture()  #hent frame fra PiCamera2, dette er array med shape: (height, width,rbg)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) #laver billed om til gråbilled

        corners, ids, _  = cv2.aruco.detectMarkers(gray, dictionary) #tjekker om vi kan finde nogle Aruco markers fra vores dictionary i billedet corners er hvor markeren er

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(  #vi beregner translation og rotation
            corners, #vi tager position fra den marker vi ønsker at kører til
            MARKER_SIZE,  #size of marker
            CAMERA_MATRIX, #camera calibration parametre
            DISTORTION_MATRIX
        )
        return ids[0], rvecs[0], tvecs[0]



    ### CALIBRATION

    def _calibrate_sleep(self, left_wheel:bool, forward_drive:bool, n:int, degrees:int, speed:int, _range=(0,1,None), wait=0.5):
        
        lower, middle, upper = _range
                
        for _ in range(n * 360//gcd(degrees, 360)):
            sleep(wait)
            arlo.go_diff(speed*left_wheel, speed*(not left_wheel), forward_drive, forward_drive)
            sleep(middle)
            arlo.stop()
                
        while True:
            userinput = input("""
Enter adjustment action [?/-/0/+/<Enter>]:
[?] Again
[-] Decrease sleep
[0] Spot on
[+] Increase sleep
<Enter> Abort
> """)
            if userinput == '?':
                pass
            elif userinput == '-':
                upper = middle
                middle = (lower+upper)/2
            elif userinput == '0':
                return {
                    "left_wheel": left_wheel,
                    "forward_drive": forward_drive,
                    "wait": wait,
                    "n": n,
                    "result": (degrees, speed, middle) # (turning degrees, wheel speed, sleep time)
                }
            elif userinput == '+':
                lower = middle
                middle = middle*2 if upper is None else (lower+upper)/2
            elif userinput == '':
                return
            else:
                print("Invalid input!")
                continue
                
            for _ in range(n * 360//gcd(degrees, 360)):
                sleep(wait)
                arlo.go_diff(speed*left_wheel, speed*(not left_wheel), forward_drive, forward_drive)
                sleep(middle)
                arlo.stop()

    def _calibrate_speed(self, left_wheel:bool, forward_drive:bool, n:int, degrees:int, _sleep:float, _range=(40,83,127), wait=0.5):
        
        # if (2 * 9.8/(3*360) * degrees) < _sleep:
        #     raise Warning("_sleep is likely to high to be satisfied by even the lowest speed.")

        lower, middle, upper = _range
                
        for _ in range(n * 360//gcd(degrees, 360)):
            sleep(wait)
            arlo.go_diff(middle*left_wheel, middle*(not left_wheel), forward_drive, forward_drive)
            sleep(_sleep)
            arlo.stop()
                
        while True:
            userinput = input("""
Enter adjustment action [?/-/0/+/<Enter>]:
[?] Again
[-] Decrease speed
[0] Spot on
[+] Increase speed
<Enter> Abort
> """)
            if userinput == '?':
                pass
            elif userinput == '-':
                upper = middle
                middle = (lower+upper)/2
            elif userinput == '0':
                return {
                    "left_wheel": left_wheel,
                    "forward_drive": forward_drive,
                    "wait": wait,
                    "n": n,
                    "result": (degrees, middle, _sleep) # (turning degrees, wheel speed, sleep time)
                }
            elif userinput == '+':
                lower = middle
                middle = middle*2 if upper is None else (lower+upper)/2
            elif userinput == '':
                return
            else:
                print("Invalid input!")
                continue
                
            for _ in range(n * 360//gcd(degrees, 360)):
                sleep(wait)
                arlo.go_diff(middle*left_wheel, middle*(not left_wheel), forward_drive, forward_drive)
                sleep(_sleep)
                arlo.stop()

    def calibration_suit(self, degrees=(2*360, 360, 180, 90, 45), speeds=(40, 50, 61, 83, 127), sleeps=(0.5, 1, 2, 4, 8), n=1, wait=0.5):
        {
            'left': {
                'forward': [],
                'bacward': [],
            },
            'right': {
                'forward': [],
                'bacward': [],
            },
        }

        for left in (True, False):
            for forward in (True, False):
                pass

    ### OBSOLETE STUFF
        
    def go(self):
        """OBSOLETE: Send a go command for continuous forward driving using the wheel encoders"""
        cmd='g\n'
        return self.send_command(cmd)
        
    def backward(self):
        """OBSOLETE: Send a backward command for continuous reverse driving using the wheel encoders"""
        cmd='v\n'
        return self.send_command(cmd)
        

    def left(self):
        """OBSOLETE: Send a rotate left command for continuous rotating left using the wheel encoders"""
        cmd='n\n'
        return self.send_command(cmd)

    def right(self):
        """OBSOLETE: Send a rotate right command for continuous rotating right using the wheel encoders"""
        cmd='m\n'
        return self.send_command(cmd)
        
    def step_forward(self):
        """OBSOLETE: Send a step forward command for driving forward using the wheel encoders for a 
           predefined amount of time"""
        cmd='f\n'
        return self.send_command(cmd)

    def step_backward(self):
        """OBSOLETE: Send a step backward command for driving backward using the wheel encoders for a 
           predefined amount of time"""
        cmd='b\n'
        return self.send_command(cmd)

    def step_rotate_left(self):
        """OBSOLETE: Send a step rotate left command for rotating left using the wheel encoders for a 
           predefined amount of time"""
        cmd='l\n'
        return self.send_command(cmd)
        
    def step_rotate_right(self):
        """OBSOLETE: Send a step rotate right command for rotating right using the wheel encoders for 
           a predefined amount of time"""
        cmd='r\n'
        return self.send_command(cmd)
            
        
    def set_speed(self, speed):
        """OBSOLETE: Speed must be a value in the range [0; 255]. This speed is used in commands based on 
           using the wheel encoders."""
        cmd='z' + str(speed) + '\n'
        return self.send_command(cmd)
        
    def set_turnspeed(self, speed):
        """OBSOLETE: Turnspeed must be a value in the range [0; 255]. This speed is used in commands based on 
           using the wheel encoders."""
        cmd='x' + str(speed) + '\n'
        return self.send_command(cmd)

    def set_step_time(self, steptime):
        """OBSOLETE: steptime is the amount of miliseconds used in the step_forward and step_backwards 
           commands."""
        cmd='t' + str(steptime) + '\n'
        return self.send_command(cmd)
        
    def set_turn_time(self, turntime):
        """OBSOLETE: turntime is the amount of miliseconds used in the step_rotate_left and 
        step_rotate_right commands."""
        cmd='y' + str(turntime) + '\n'
        return self.send_command(cmd)
        
arlo = Robot()
