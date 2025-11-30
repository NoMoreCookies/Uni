import mnist_loader
from Neural_Network import Network 

network = Network([784, 40, 30, 10])
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

network.StochasticGradientDescent(training_data, 50, 5, 1, test_data=test_data)



