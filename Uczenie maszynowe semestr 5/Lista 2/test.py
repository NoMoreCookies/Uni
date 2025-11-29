import mnist_loader
from Neural_Network import Network 

net = Network([784,30,10])
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net.SGD(training_data,30,10,3,test_data = test_data)



