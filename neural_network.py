"""
A feed forward neural network with one layer
"""

# numpy imports 
from numpy import dot, exp, array, random

# neural network class 
class NeuralNetwork(object):
    
    # init function for inital weights 
    def __init__(self):
        # setting random seed 
        random.seed(1)

        # intial weights 
        self.synaptic_weights = 2 * random.random((3,1)) - 1

    # making an activation function 
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # making a sigmoid derivative 
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # making a training function 
    def train(self, training_set_input, training_set_output, num_iterations):
        for iterations in range(num_iterations):

            # computing outputs 
            outputs = self.predict(training_set_input)

            # error values 
            error = training_set_output - outputs

            # adjustment value for weights using back prop 
            adjustment = dot(training_set_input.T, error * self.__sigmoid_derivative(outputs))

            # adjusting weights according to error 
            self.synaptic_weights += adjustment

    # making a predict function 
    def predict(self, input_value):
        return self.__sigmoid(dot(input_value, self.synaptic_weights))


def main():

    # making a NN object 
    neural_network = NeuralNetwork()

    # intial weights 
    print("Initial Synaptics weights")
    print(neural_network.synaptic_weights)

    # data set 
    training_set_input = array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
    training_set_output = array([[0,1,1,0]]).T

    # train nn 
    neural_network.train(training_set_input, training_set_output, 10000)

    # printing after train synaptic weights 
    print("After Training synaptics weights ")
    print(neural_network.synaptic_weights)

    # predicting for [1,0,0]
    print(neural_network.predict(array([1,0,0])))

if __name__ == '__main__':
    main()