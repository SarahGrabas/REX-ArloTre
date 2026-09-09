# from time import sleep
# import robot
import numpy as np
import matplotlib.pyplot as plt

# arlo = robot.Robot()

# def measure_distance(repeats=5):
#     measurements = []
#     for i in range(repeats):
#         distance_mm = arlo.read_front_ping_sensor()
#         measurements.append(distance_mm)
#         print("Measurement", i + 1, "=", distance_mm, "mm")
#         sleep(0.1)
#     return measurements

# results = measure_distance()
# print("All measurements:", results)


# Sonar-sensordata mod træplade:
# All measurements, 100 mm: [92, 93, 93, 92, 92]
# All measurements, 200 mm: [193, 193, 193, 193, 193]
# All measurements, 500 mm: [486, 487, 486, 486, 486]
# // Har fjernet denne // All measurements, 800 mm: [788, 786, 786, 786, 785]
# All measurements, 1000 mm: [987, 988, 988, 988, 988]
# All measurements, 2000 mm: [2000, 1995, 2000, 2000, 1996]


# Vi bør fjerne en af dem. Vi har 6 afstande.
true_distance = np.array([100, 200, 500, 1000, 2000])
# [788, 786, 786, 786, 785],


measurements = np.array([
    [92, 93, 93, 92, 92],
    [193, 193, 193, 193, 193],
    [486, 487, 486, 486, 486],
    [987, 988, 988, 988, 988],
    [2000, 1995, 2000, 2000, 1996]
])

# true_distance_cm = true_distance / 10
# measurements_cm = measurements / 10

errors = measurements - true_distance[:, None]
mean_measurement = np.mean(measurements, axis=1)
std_error = np.std(errors, axis=1, ddof=1)


print("Mean measurements [mm]:", mean_measurement)
print("Standard deviations [mm]:", std_error)

plt.figure()
plt.scatter(true_distance, std_error)
plt.xlabel("True distance [mm]")
plt.ylabel("Standard deviation [mm]")
plt.title("Sonar precision as a function of distance")
plt.grid()
plt.show()



# Linearity plot
# Regressionsdel
slope, intercept = np.polyfit(true_distance, mean_measurement,1)

linear_fit = slope * true_distance + intercept

# Starter med at beregne R^^2
ss_res = np.sum((mean_measurement - linear_fit) ** 2)
ss_tot = np.sum((mean_measurement - np.mean(mean_measurement)) ** 2) 
r_squared = 1 - ss_res / ss_tot

print("Mean measurements [mm]:", mean_measurement)
print("Standard deviations [mm]:", std_error)
print("Slope:", slope)
print("Intercept [mm]:", intercept)
print("R²:", r_squared)

plt.figure()

# Every measurement as a light blue dot - de ligger bag ved mean measurements 
for i, distance in enumerate(true_distance):
    plt.scatter(
        np.full(measurements.shape[1], distance),
        measurements[i],
        color="lightblue"
    )

# Mean measurements as a blue dot
plt.scatter(
    true_distance,
    mean_measurement,
    color="blue",
    label="Mean measurements"
)

# Best fit line 
plt.plot(
    true_distance,
    linear_fit,
    color="red",
    label=f"Linear fit, R² = {r_squared:.5f}"
)

# Ideel sammenhæng
plt.plot(
    true_distance,
    true_distance,
    "--",
    color="black",
    label="Ideal: measured = true distance"
)

plt.xlabel("True distance [mm]")
plt.ylabel("Measured distance [mm]")
plt.title("Measured distance as a function of true distance")
plt.legend()
plt.grid()
plt.show()
