import mnist_second as ms
import heart_disease_loader as hdl
from neural_nets import *

"""
This file aims to illustrate the way Network class works
change if False to if True if you want to test particular
bit of code
"""

if False:
    """
    MNIST dataset
    baseline neural net with sigmoids in hidden layers
    and softmax in the output layer
    """
    train_data, test_data = ms.load_mnist_wrapper2()
    n = Network([784, 100, 10], Entropy, Sigmoid, Sigmoid, True)
    n.train_network(train_data, 30, 10, 0.5, 1, test_data[400:])

if True:
    """
    MNIST dataset
    modified neural net with cosine/sine activation functions
    in hidden layers and softmax in the output layer
    No bias term for any score apart from the one in the first layer
    """
    train_data, test_data = ms.load_mnist_wrapper2()
    n = Network([784, 100, 10], Entropy, Cos, Sin, False)
    n.train_network(train_data, 30, 10, 0.5, 1, test_data[400:])

if False:
    """
    South African Heart Disease dataset
    modified neural net with cosine/sine activation functions
    in hidden layers and softmax in the output layer
    No bias term for any score apart from the one in the first layer
    """
    data = hdl.get_heart_disease_data()
    train_data = data[:400]
    test_data = data[400:]
    n = Network([9, 20, 2], Entropy, Cos, Sin, False)
    n.train_network(train_data, 30, 10, 0.1, 10, test_data)
