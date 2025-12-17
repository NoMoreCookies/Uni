import mnist_loader
from Neural_Network import Network 

network = Network([784, 128, 64, 10])
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

network.StochasticGradientDescent(training_data, 10, 10, 1, test_data=test_data)



