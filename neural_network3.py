"""
A Feed forward neural network with three layers 
"""

# numpy imports 
from numpy import dot, exp, array, random

# making a NN Class 
class NeuralNetwork(object):

    # init constructor
    def __init__(self, layer_2 = 5, layer_3 = 4):
            # making a random seed 
            random.seed(1)

            # initializing weights 
            self.synaptics_weights1 = 2 * random.random((3, layer_2)) - 1 # input layer weights
            self.synaptics_weights2 = 2 * random.random((layer_2,layer_3)) - 1 # hidden layer weights
            self.synaptics_weights3 = 2 * random.random((layer_3, 1)) - 1 # output layer weights

    # make sigmoid function 
    # an activation function 
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # derivative of sigmoid function 
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # making a training function 
    def train(self, training_set_inputs, training_set_outputs, num_iterations):
        for iterations in range(num_iterations):

            # getting output values 
            activation_layer1 = self.__sigmoid(dot(training_set_inputs, self.synaptics_weights1))
            activation_layer2 = self.__sigmoid(dot(activation_layer1, self.synaptics_weights2))
            output = self.__sigmoid(dot(activation_layer2, self.synaptics_weights3))

            # computing errors 
            del4 = (training_set_outputs - output) * self.__sigmoid_derivative(output) # output layer

            # computing errors for each layer in reverse order
            del3 = dot(self.synaptics_weights3, del4.T) * (self.__sigmoid_derivative(activation_layer2).T) # hidden layer
            del2 = dot(self.synaptics_weights2, del3) * (self.__sigmoid_derivative(activation_layer1).T) # input later

            # getting adjustments
            # back propogation in reverse order 
            adjustment3 = dot(activation_layer2.T, del4) # output layer
            adjustment2 = dot(activation_layer1.T, del3.T) # hidden layer
            adjustment1 = dot(training_set_inputs.T, del2.T) # input layer 

            # adjusting synaptic weights 
            self.synaptics_weights1 += adjustment1 # input layer 
            self.synaptics_weights2 += adjustment2 # hidden layer 
            self.synaptics_weights3 += adjustment3 # output layer 

    # making a prediction function
    def predict(self, input_value):
        # input to layer 1 
        activation_layer1 = self.__sigmoid(dot(input_value, self.synaptics_weights1))
        activation_layer2 = self.__sigmoid(dot(activation_layer1, self.synaptics_weights2)) # hidden layer 
        output = self.__sigmoid(dot(activation_layer2, self.synaptics_weights3)) # output layer 

        return output # return output 

def main():
    # Getting inputs of layers 
    layer2 = int(input("Enter no of neurons in layer 2 "))
    layer3 = int(input("Enter no of neurons in layer 3 "))

    # making a NN obj 
    neural_network = NeuralNetwork(layer2, layer3)

    # printing inital synaptic weights 
    print("Initial Value of Synaptic weights of input layer is ")
    print(neural_network.synaptics_weights1)
    print("Initial Value of Synaptic weights of hidden layer is ")
    print(neural_network.synaptics_weights2)
    print("Initial Value of Synaptic weights of output layer is ")
    print(neural_network.synaptics_weights3)

    # making training data set 
    training_set_input = array([[0,0,1], [1,1,1],[1,0,1], [0,1,1]])
    training_set_output = array([[0,1,1,0]]).T

    # train our NN 
    neural_network.train(training_set_input, training_set_output, 10000)

    # printing synaptics weights after training of NN
    print("Value of Synaptic weights of input layer after training is ")
    print(neural_network.synaptics_weights1)
    print("Value of Synaptic weights of hidden layer after training is ")
    print(neural_network.synaptics_weights2)
    print("Value of Synaptic weights of output layer after training is ")
    print(neural_network.synaptics_weights3)

    # predicting a value of any random input 
    print("Value of input [1,0,0] is ")
    print(neural_network.predict(array([1,0,0])))

if __name__ == '__main__':
    main()