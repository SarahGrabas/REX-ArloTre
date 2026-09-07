from time import sleep
import robot

arlo = robot.Robot()


def measure_distance(repeats=5):
    measurements = []
    for i in range(repeats):
        distance_mm = arlo.read_front_ping_sensor()
        measurements.append(distance_mm)
        print("Measurement", i + 1, "=", distance_mm, "mm")
        sleep(0.1)
    return measurements

results = measure_distance()
print("All measurements:", results)
