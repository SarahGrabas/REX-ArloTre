import numpy as np


def sample_motion_model_velocity(particles, velocity, angular_velocity, delta_t):

    # a1: Hvor meget usikkerhed der kommer på translationshastigheden,
    #     når robotten kører fremad. Større a1 betyder større variation
    #     i hvor langt hver particle bevæger sig.
    #
    # a2: Hvor meget robotrotation bidrager til usikkerhed i
    #     translationshastigheden. Hvis robotten både kører og drejer,
    #     kan rotationen derfor give ekstra usikkerhed i positionen.
    #
    # a3: Hvor meget robotternes translation bidrager til usikkerhed
    #     i rotationshastigheden. Større a3 giver mere variation
    #     i hvor meget robotten drejer, når den kører.
    #
    # a4: Hvor meget selve rotationen bidrager til usikkerhed i
    #     rotationshastigheden. Større a4 betyder større variation
    #     i den estimerede drejevinkel ved rotation.
    #
    # a5: Ekstra støj på den endelige orientering, som afhænger af
    #     robotternes translationshastighed. Bruges til at modellere,
    #     at kørsel også kan påvirke robotens orientering.
    #
    # a6: Ekstra støj på den endelige orientering, som afhænger af
    #     robotternes rotationshastighed. Større a6 giver mere usikkerhed
    #     i orienteringen efter en rotation.
    
    a1 = 0.40
    a2 = 0.20
    a3 = 0.20
    a4 = 0.20
    a5 = 0.10
    a6 = 0.20

    for p in particles:

        # Hvis robotten roterer på stedet,
        # skal translationshastigheden være 0
        if abs(velocity) < 1e-6:

            v_hat = 0.0

        else:

            v_hat = velocity + np.random.normal(
                0,
                np.sqrt(
                    a1 * velocity**2 +
                    a2 * angular_velocity**2
                )
            )

        w_hat = angular_velocity + np.random.normal(
            0,
            np.sqrt(
                a3 * velocity**2 +
                a4 * angular_velocity**2
            )
        )

        gamma_hat = np.random.normal(
            0,
            np.sqrt(
                a5 * velocity**2 +
                a6 * angular_velocity**2
            )
        )

        x = p.getX()
        y = p.getY()
        theta = p.getTheta()

        x_new = x + v_hat * delta_t * np.cos(theta)
        y_new = y + v_hat * delta_t * np.sin(theta)

        theta_new = np.mod(
            theta +
            w_hat * delta_t +
            gamma_hat * delta_t,
            2.0 * np.pi
        )

        p.setX(x_new)
        p.setY(y_new)
        p.setTheta(theta_new)


# def sample_motion_model_velocity(particles, velocity, angular_velocity, delta_t):

#     # a1 = 0.05  # Translationsfejl fra translation (cm fejl per cm kørt)
#     # a2 = 0.01  # Translationsfejl fra rotation
#     # a3 = 0.01  # Rotationsfejl fra translation
#     # a4 = 0.05  # Rotationsfejl fra rotation (rad fejl per rad drejet)
#     # a5 = 0.01  # Ekstra slutrotationsstøj fra translation
#     # a6 = 0.01  # Ekstra slutrotationsstøj fra rotation

#     a1 = 0.02
#     a2 = 0.005
#     a3 = 0.001
#     a4 = 0.01
#     a5 = 0.001
#     a6 = 0.001

#     # Prediktionstrin (sample_motion_model_velocity) for alle partikler
#     for p in particles:
#         # Træk støjbehæftet styring
#         v_hat = velocity + np.random.normal(0, np.sqrt(a1 * velocity**2 + a2 * angular_velocity**2))
#         w_hat = angular_velocity + np.random.normal(0, np.sqrt(a3 * velocity**2 + a4 * angular_velocity**2))
#         gamma_hat = np.random.normal(0, np.sqrt(a5 * velocity**2 + a6 * angular_velocity**2))

#         # Hent partiklens nuværende tilstand
#         x = p.getX()
#         y = p.getY()
#         theta = p.getTheta()

#         # Opdater tilstand
#         x_new = x + v_hat * delta_t * np.cos(theta)
#         y_new = y + v_hat * delta_t * np.sin(theta)
        
#         theta_new = np.mod(theta + w_hat * delta_t + gamma_hat * delta_t, 2.0 * np.pi)

#         # Gem ny tilstand i partiklen
#         p.setX(x_new)
#         p.setY(y_new)
#         p.setTheta(theta_new)
