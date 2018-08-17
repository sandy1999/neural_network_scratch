# numpy imports 
from numpy import dot, array, exp, random

# NN class 
class NeuralNetwork(object):
    def __init__(self, layer_2 = 5, layer_3 = 4, layer_4 = 5):
            # random seed 
            random.seed(1)

            # initial synaptic weights 
            self.synaptic_weights1 = 2 * random.random((3, layer_2)) - 1 # input layer
            self.synaptic_weights2 = 2 * random.random((layer_2, layer_3)) - 1 # hidden layer 1
            self.synaptic_weights3 = 2 * random.random((layer_3, layer_4)) - 1 # hidden layer 2
            self.synaptic_weights4 = 2 * random.random((layer_4, 1)) - 1 # output layer

    # making a sigmoid function 
    # activation function adjust value between 0 and 1
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))
    
    # making a sigmoid derivative 
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_set_inputs,training_set_outputs, num_iterations):
        for iteration in range(num_iterations):

            # getting output 
            activation_layer1 = self.__sigmoid(dot(training_set_inputs, self.synaptic_weights1)) # input layer 
            activation_layer2 = self.__sigmoid(dot(activation_layer1, self.synaptic_weights2)) # hidden layer 1
            activation_layer3 = self.__sigmoid(dot(activation_layer2, self.synaptic_weights3)) # hiddeen layer 2
            output = self.__sigmoid(dot(activation_layer3, self.synaptic_weights4)) # output layer

            # computing error 
            del5 = (training_set_outputs - output) * self.__sigmoid_derivative(output) # output layer

            # computing error for each layer
            del4 = dot(self.synaptic_weights4, del5.T) * (self.__sigmoid_derivative(activation_layer3).T) # hidden layer 2
            del3 = dot(self.synaptic_weights3, del4) * (self.__sigmoid_derivative(activation_layer2).T) # hidden layer 1
            del2 = dot(self.synaptic_weights2,del3) * (self.__sigmoid_derivative(activation_layer1).T) # input layer

            # getting adjustments
            adjustment4 = dot(activation_layer3.T, del5) # output layer 
            adjustment3 = dot(activation_layer2.T, del4.T) # hidden layer 2
            adjustment2 = dot(activation_layer1.T, del3.T) # hidden layer 1
            adjustment1 = dot(training_set_inputs.T, del2.T) # input layer

            # adjusting weights
            self.synaptic_weights1 += adjustment1 # input layer
            self.synaptic_weights2 += adjustment2 # hidden layer 1
            self.synaptic_weights3 += adjustment3 # hidden layer 2
            self.synaptic_weights4 += adjustment4 # output layer
    # prediction method
    def predict(self, input_value):
        activation_layer1 = self.__sigmoid(dot(input_value, self.synaptic_weights1)) # input layer 
        activation_layer2 = self.__sigmoid(dot(activation_layer1, self.synaptic_weights2)) # hidden layer 1
        activation_layer3 = self.__sigmoid(dot(activation_layer2, self.synaptic_weights3)) # hiddeen layer 2
        output = self.__sigmoid(dot(activation_layer3, self.synaptic_weights4)) # output layer

        return output

def main():
    # taking input neurons
    layer2 = int(input("Enter number of neuron for layer 2 "))
    layer3 = int(input("Enter number of neuron for layer 3 "))
    layer4 = int(input("Enter number of neuron for layer 4 "))
    
    neural_network = NeuralNetwork(layer2, layer3, layer4)
    # print initial synaptic weights
    print("Inital Value of Synaptic Weights of input layer are:")
    print(neural_network.synaptic_weights1)
    print("Inital Value of Synaptic Weights of hidden layer 1 are:")
    print(neural_network.synaptic_weights2)
    print("Inital Value of Synaptic Weights of hidden layer 2 are:")
    print(neural_network.synaptic_weights3)
    print("Inital Value of Synaptic Weights of output layer are:")
    print(neural_network.synaptic_weights4)

    # making data set 
    training_set_input = array([[0,0,1], [1,1,1], [1,0,1],[0,1,1]])
    training_set_output = array([[0,1,1,0]]).T

    # training neural network
    neural_network.train(training_set_input, training_set_output, 10000)

    # printing value of trained weights
    print("Value of Synaptic Weights of input layer after training are:")
    print(neural_network.synaptic_weights1)
    print("Value of Synaptic Weights of hidden layer 1 after training are:")
    print(neural_network.synaptic_weights2)
    print("Value of Synaptic Weights of hidden layer 2 after training are:")
    print(neural_network.synaptic_weights3)
    print("Value of Synaptic Weights of output layer after training are:")
    print(neural_network.synaptic_weights4)

    # predicting test value
    print("Value of [1,0,0] is: ")
    print(neural_network.predict(array([1,0,0])))

if __name__ == '__main__':
    main()