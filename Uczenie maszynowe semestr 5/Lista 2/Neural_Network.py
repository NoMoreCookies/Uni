import numpy as np
import random


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
        self.biases = [np.random.randn(y, 1)
                       for y in sizes[1:]]
        
        # random generated weights
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
           
    def feedforward(self, a):
        """ FeedForward func 

        It is responsible for calculating actiations for 
        next neureons in our neural network

        Args:
            self (Network) : Object
            a (float list) : Outputs fromm last layers
        """
        for b, w in zip(self.biases, self.weights):

            # calculating with sigmoid function
            a = sigmoid(np.dot(w, a)+b)

        return a
        
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Stochastic Gradient Descent
        
        Finds local minima for weights and biases of our neural network 
        using Stochastic Gradient Descent Algorithm

        Args:
            self (Network) : Object
            training_data (list of pairs) : Image and number on image
            epochs (int) :  Number of learning steps of our network
            mini_batch_size (int) :  Length of batches that training data should be divied into
            eta (float) : Length of our SGD step 
            test_data (list of pairs) : Image and number on image
        
        """

        if test_data:
            n_test = len(test_data)

        n = len(training_data)

        for j in range(epochs):

            #random training data shuffle
            random.shuffle(training_data)

            # rendering list of mini batches
            mini_batches = [
                training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size )
                ] 
            
            # teaching our nneural network on with every batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            # evaluating performance of our neural network
            # after previous learning step
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data),n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):

        # dla każdej warstwy w każdej sieci są biases
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        # dla każdej warstwy w każdej sieci są biases
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            # propagacja wsteczna
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)


            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]


            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # update biasów i wag

        self.weights = [w - (eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        
        self.biases = [b - (eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
    
    def backprop(self, x, y):

        # tworzymy wetkroy zer dla b 
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        # tworzymy wetkroy zer dla b 
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # aktywacje
        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.biases, self.weights):
            # przemnazamy aktywacje prze wagi i dodajemy bias
            z = np.dot(w,activation)+b
            zs.append(z)

            # przepuszczamy przez funckje sigmoid
            activation = sigmoid(z)
            activations.append(activation)

        
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1] )

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(),delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) 
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    # funckja aktywacyjna
    def cost_derivative(self, output_activations, y):
        return (output_activations - y)
    
# funcjja sigmoid
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

# jakis dziwny sigmoid, nie rozumiem go
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
