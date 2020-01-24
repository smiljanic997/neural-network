import numpy as np
from random import shuffle


def main():
    # res = generate_training_examples()
    # count = 0
    # with open('training2.txt', 'w') as f:
    #     for pair in res:
    #         if test(pair[3], pair[2], pair[0], pair[1]):
    #             count = count + 1
    #             np.savetxt(f, pair, delimiter = ',', fmt = '%1.15f')
    res = generate_test_examples()
    with open('test.txt', 'w') as f:
        for pair in res:
            np.savetxt(f, pair, delimiter = ',', fmt = '%1.15f')

def generate_training_examples():
    distances = np.arange(6.75, 18, 0.35)
    shuffle(distances)
    def_distances = np.arange(1, 3, 0.15)
    shuffle(def_distances)
    angles = []
    speeds = []
    result = []
    for def_distance in def_distances:
        angles.append(np.arctan(0.4 / def_distance) + np.radians(25))
    for angle, def_distance in zip(angles, def_distances):
        for distance in distances:
            speed = get_speed(distance, 0.55, angle)
            result.append([def_distance, distance, angle, speed])
    return result

def generate_test_examples():
    distances = np.random.uniform(low = 6.75, high = 18.0, size=(500,))
    shuffle(distances)
    def_distances = np.random.uniform(low = 1.0, high = 3.0, size=(50,))
    shuffle(def_distances)
    angles = []
    speeds = []
    result = []
    for def_distance in def_distances:
        angles.append(np.arctan(0.4 / def_distance) + np.radians(25))
    for angle, def_distance in zip(angles, def_distances):
        for distance in distances:
            speed = get_speed(distance, 0.55, angle)
            result.append([def_distance, distance, angle, speed])
    return result

def projectile_motion(x, speed, theta):
    y = np.tan(theta) * x - (9.81 * x ** 2 / (2 * (speed ** 2) * (np.cos(theta) ** 2)))
    return y

def get_speed(x, y, theta):
    g = 9.81 * x**2
    d = 2 * np.square(np.cos(theta)) * (np.tan(theta) * x - y)
    return np.sqrt(g / d)

def test(speed, angle, def_dist, distance):
    y = projectile_motion(def_dist, speed, angle)
    if y <= 0.4:
        return False # fail - block

    r_ball = 0.24 / 2
    lower = distance - r_ball
    upper = distance + r_ball
    to_check = np.arange(lower, upper, 0.01)
    return any(y == 0.55 for y in apply_pm_equation(to_check, speed, angle))

def apply_pm_equation(to_check, speed, angle):
    ret_values = []
    for d in to_check:
        ret_values.append(round(projectile_motion(d, speed, angle), 2))
    return ret_values

if __name__=='__main__':
    main()