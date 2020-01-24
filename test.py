import numpy as np
from network_cpy import Network
from activation import sigmoid, dsigmoid
from time import sleep
from multiprocessing import Process
import pandas as pd
import os
import time


def projectile_motion(x, speed, theta):
    y = np.tan(theta) * x - (9.81 * x ** 2 / (2 * (speed ** 2) * (np.cos(theta) ** 2)))
    return y

def get_speed(x, y, theta):
    g = 9.81 * x**2
    d = 2 * np.square(np.cos(theta)) * (np.tan(theta) * x - y)
    return np.sqrt(g / d)

def get_upper_x(y, speed, theta):
    c1 = (-2 / 9.81) * speed**2 * np.cos(theta)**2 * np.tan(theta)
    c2 = (2 / 9.81) * speed**2 * np.cos(theta)**2 * y
    return round(np.max(np.roots([1, c1, c2])), 2)

def test(speed, angle, def_dist, distance):
    y = projectile_motion(def_dist, speed, angle)
    if y <= 0.4:
        return False # fail - block
  
    r_ball = 0.24 / 2
    lower = distance - r_ball
    upper = distance + r_ball
    to_check = np.arange(lower, upper, 0.001)
    return any(y == 0.55 for y in apply_pm_equation(to_check, speed, angle))

def apply_pm_equation(to_check, speed, angle):
    ret_values = []
    for d in to_check:
        ret_values.append(round(projectile_motion(d, speed, angle), 2))
    return ret_values

def load_training_examples(path):
    inputs = []
    outputs = []
    data = np.loadtxt(path)
    for i in range(len(data)):
        if i % 4 == 0:
            inputs.append(data[i])
        elif i % 4 == 1:
            inputs.append(data[i])
        elif i % 4 == 2:
            outputs.append(data[i])
        elif i % 4 == 3:
            outputs.append(data[i])
    return np.split(np.array(inputs), len(data) / 4), np.split(np.array(outputs), len(data) / 4)



nn = Network(number_of_inputs = 2,
            number_of_outputs = 2,
            hidden_layer_count = 46,
            learning_rate = 0.011)
# inputs, outputs = load_training_examples('training2.txt')
inputs1, dummy = load_training_examples('test.txt')
nn.load_weights_and_biases()
# start = time.time()
# nn.train(inputs, outputs)
# nn.write_weights_to_csv()
# print(round(time.time() - start, 2) / 60, 'min')


print('Testing on 25k test data\n........\n')
scored = 0
started = time.time()
for inp in inputs1:
    g = nn.feedforward(inp)[0]
    angle = g[0][0]
    speed = g[1][0]
    def_dist = inp[0]
    dist = inp[1]
    if test(speed, angle, def_dist, dist):
        scored = scored + 1
print('Shooting took ',round((time.time() - started) / 60, 2), 'min')
print('Result:', scored, '/ 25000 ->', scored / len(inputs1) * 100, '%')



print('\nUser-defined values: ')
again = '1'
while again == '1':
    in1 = float(input('\nDefender distance: '))
    in2 = float(input('Distance from the basket: '))
    out = nn.feedforward([in1, in2])[0]
    print('\nAngle[degrees]:',np.degrees(out[0]),'\nSpeed:',out[1])
    if test(out[1][0], out[0][0], in1, in2):
        print('\nHIT!')
    else:
        print('\nMISS!')
    again = input('Try again ? [0/1] ')    
