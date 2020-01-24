# Neural network - Python implementation

Problem: train a network so that it calculates angle and
initial speed of a basketball so that the basket is scored
each time.
The shots are taken from 6.75m to 18m distance. There is also
a defender trying to block the shot. Defender is 1m to 3m
far from the shot place.


'test.py' script shall be run in order to start the shooting.
In this implementation, NN tries 25000 shots(test data).
The network itself, along with backpropagation algorithm,
is implemented in 'network_cpy.py' file. It's already trained
and the weights are saved as the csv files in 'weights' folder.

Training(and test) data is generated using 'generator.py' script.
