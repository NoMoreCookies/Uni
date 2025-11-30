import numpy as np
import random

# Kod został napisany na podstawie książki "Neural Networks and Deep Learning"
# Michaela Nielsena, 
# Wzory w niej przedstawione są zgodne z wykładowymi, z tą różnicą, że tutaj
# mamy biasy oddzielnie, gdzie w wykładach są one w macierzach wag


class Network(object):
    def __init__(self, sizes):
        """Initializes our Network object

        Initializes Network object with given sizs of hidden layers

        Args:
            sizes (list of ints)  - lengths of hiddens layers
        """

        # number of hidden layers
        self.num_layers = len(sizes)

        # random generated biases
        # (from sizes[1] because input layer has no biases)
        self.biases = [np.random.randn(y, 1)
                       for y in sizes[1:]]
             
        # random generated weights (shift is because weights are between
        # layers and we need matrix multiplication for forward propagation)
        self.weights = [np.random.randn(y, a)
                        for a, y in zip(sizes[:-1], sizes[1:])]
           
    def feedForward(self, a):
        """ FeedForward function

        It is responsible for calculating activations for
        next neureons in our neural network

        Args:
            self (Network) : Object
            a (float list) : Outputs fromm last layers
        """
        for bias, weight in zip(self.biases, self.weights):

            # calculating with sigmoid function
            # (sigmoid returns numbers bewteen 0 and 1)
            # 0 when numbers are very negative
            # and 1 when numbers are very positive
            a = sigmoid(np.dot(weight, a)+bias)

        return a
        
    def StochasticGradientDescent(self, training_data, epochs,
                                    trainSampleSize, eta, test_data=None):
        """Stochastic Gradient Descent
        
        Finds local minima for weights and biases of our neural network
        using Stochastic Gradient Descent Algorithm

        Args:
            self (Network) : Object
            training_data (list of pairs) : Image and number on image
            epochs (int) :  Number of learning steps of our network
            trainSampleSize (int) :  Length of batches that
            training data should be divied into
            eta (float) : Length of our StochasticGradientDescent step
            test_data (list of pairs) : Image and number on image
        
        """

        n = len(training_data)

        for j in range(epochs):

            # random training data shuffle
            random.shuffle(training_data)

            # rendering list of mini batches
            miniSamples = [
                training_data[k: k+trainSampleSize] for
                k in range(0, n, trainSampleSize)
                ]
            
            # teaching our neural network with every training sample
            for mini_sample in miniSamples:
                self.updateBasedOnMiniSample(mini_sample, eta)

            # evaluating performance of our neural network
            # after previous learning step
            if test_data:
                print("Step {0}: score {1} ".format(j, self.score(test_data)))
            else:
                print("Step number: {0} completed".format(j))

    def updateBasedOnMiniSample(self, mini_sample, eta):
        """"Updates Based On Mini Sample
        
        Updates weights and biases of our neural network
        using gradient descent from backpropagation
        
        Args:
            self (Network) : Object
            mini_sample (list of pairs) : Image and number on image
            eta (float) : Length of our StochasticGradientDescent step
        
        """

        # for each layer of biases we create zero vectors
        dC_db = [np.zeros(b.shape) for b in self.biases]

        # for each layer of weights we create zero matrices
        dC_dw = [np.zeros(w.shape) for w in self.weights]

        for a, y in mini_sample:

            # backpropagation
            delta_dC_db, delta_dC_dw = self.backPropagation(a, y)

            # sum of gradients from all samples in mini batch
            dC_db = [nb + dnb for nb, dnb in zip(dC_db, delta_dC_db)]

            dC_dw = [nw + dnw for nw, dnw in zip(dC_dw, delta_dC_dw)]

        # updating weights and biases using mean from mini batch gradients
        self.weights = [weight - (eta/len(mini_sample))*nweight
                        for weight, nweight in zip(self.weights, dC_dw)]
        
        self.biases = [bias - (eta/len(mini_sample))*nbias
                       for bias, nbias in zip(self.biases, dC_db)]
    
    def backPropagation(self, x, y):
        """Backpropagation

        Calculates gradient of cost_derivative function with respect to
        weights and biases using backpropagation algorithm

        Args:
            self (Network) : Object
            x (float list) : Input image
            y (float list) : Expected output
        """

        # vekctors of zeros for biases
        dC_db = [np.zeros(b.shape) for b in self.biases]

        # matrices of zeros for weights
        dC_dw = [np.zeros(w.shape) for w in self.weights]

        # activations
        activation = x

        # we need to remember all activations  for gradient descent
        activations = [x]
        
        # outputs of all layers before activation function
        z_layers = []

        for bias, weight in zip(self.biases, self.weights):

            # we calculate layers before activation
            z = np.dot(weight, activation) + bias
            z_layers.append(z)

            # we put it through sigmoid function
            activation = sigmoid(z)

            # we store it to activations
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) *\
            sigmoid_derivative(z_layers[-1])

        # we calculate new weights and biases for last layer
        dC_db[-1] = delta
        dC_dw[-1] = np.dot(delta, activations[-2].transpose())
 
        # now we need to calculate them for inner layers,
        # we use chain formula to get the derivatives
        for l in range(2, self.num_layers):

            # we start from second to last layer
            z = z_layers[-l]

            # we calc derative of sigmoid function from outputs before activation
            sigm_der = sigmoid_derivative(z)

            # we calculate delta for current layer
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sigm_der

            # we update biases and weights gradients
            dC_db[-l] = delta

            dC_dw[-l] = np.dot(delta, activations[-l-1].transpose())

        return (dC_db, dC_dw)
    
    def score(self, test_data):
        """Evaluate neural network

        Evaluates performance of our neural network based on the test data

        Args:
            self (Network) : Object
            test_data (list of pairs) : Image and number on image

        """

        test_results = [(np.argmax(self.feedForward(a)), y) 
                        for (a, y) in test_data]
        
        return sum(int(y_predicted == y_true) for (y_predicted, y_true) in
                    test_results)
    
    def cost_derivative(self, output_activations, y):
        """Cost Derivative
        Args:
            self (Network) : Object
            output_activations (float list) : Outputs from neural network
            y (float list) : Expected outputs

        returns:
            float list : Derative of cost_derivative function
        """
        return (output_activations - y)
    

def sigmoid(z):
    """"Sigmoid function

    Args:
        z (float) : Input number

    returns:
        float : Sigmoid of input number

    """
    return 1.0/(1.0+np.exp(-z))


def sigmoid_derivative(z):
    """Derative of Sigmoid function

    Args:
        z (float) : Input number

    returns:
        float : Derative of Sigmoid function
    """
    return sigmoid(z)*(1-sigmoid(z))
