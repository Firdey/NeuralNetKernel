import loaders.load_MNIST as ms
import loaders.load_heart_disease as hd
import loaders.load_MISR as misr
import loaders.load_Adult as ad
import src.neural_net as nn
import src.neural_net_ext as nne
import src.randff as rff
import src.randff_ext as rffe
import numpy as np

"""
This file aims to test various bits of codes for Module 5
change "if False" to "if True" for testing particular
bits of code.
"""

if False:
    """
    test of neural_net
    
    MNIST dataset
    baseline neural net with sigmoids in hidden layers
    and softmax in the output layer
    """
    train_data, test_data = ms.load_mnist_wrapper2()
    n = nn.Network([784, 100, 10], nn.Entropy, nn.Sigmoid, nn.Sigmoid, True)
    n.train_network(train_data, 30, 10, 0.5, 1, test_data)

if False:
    """
    test of neural_net
    
    MNIST dataset
    modified neural net with cosine/sine activation functions
    in hidden layers and softmax in the output layer
    No bias term for any score apart from the one in the first layer
    """
    train_data, test_data = ms.load_mnist_wrapper2()
    n = nn.Network([784, 40, 10], nn.Entropy, nn.Cos, nn.Sin, False)
    n.train_network(train_data, 30, 10, 0.5, 5, test_data)

if False:
    """
    test of neural_net
    
    South African Heart Disease dataset
    modified neural net with cosine/sine activation functions
    in hidden layers and softmax in the output layer
    No bias term for any score apart from the one in the first layer
    """
    data = hd.get_heart_disease_data()
    np.random.shuffle(data)
    train_data = data[:400]
    test_data = data[400:]
    n = Network([9, 20, 2], nn.Entropy, nn.Cos, nn.Sin, False)
    n.train_network(train_data, 30, 10, 0.1, 10, test_data)

if False:
    """
    test of neural_net

    Adult dataset
    modified neural net with cosine/sine activation functions
    in hidden layers and softmax in the output layer
    No bias term for any score apart from the one in the first layer
    """
    data_train, data_test = ad.load_Adult_wrapper()
    net = nn.Network([108, 500, 2], nn.Entropy, nn.Cos, nn.Sin, False)
    net.train_network(data_train, 30, 10, 0.1, 1, data_test)

if False:
    """
    test of neural_net_ext
    
    MISR dataset
    modified neural net with three layer + mean pooling structure:
    hidden cos/sin || mean pooling || hidden cos/sin || output layer
    no bias term beyond first layer
    """
    data = misr.load_MISR()
    np.random.shuffle(data)
    train_data = data[:700]
    test_data = data[700:]
    net = nne.Network([16,20,20,1], 100, nne.Quad_reg, nne.Cos, nne.Sin)
    net.train_network(train_data, 30, 10, 1, 0.1, test_data)

if False:
    """
    test of randomff
    
    Adult dataset
    Random Fourier features with a single kernel
    """
    data_train, data_test = ad.load_Adult_wrapper(2)
    X = data_train[0]
    y = data_train[1]
    X_te = data_test[0]
    y_te = data_test[1]
    rff_obj = rff.randomff(D = 500, p = 108, lamda = 0.01, sigma = 0.5)
    rff_obj.train(X,y)
    print(rff_obj.test(X_te,y_te))

if False:
    """
    test of randomff_ext
    
    MISR dataset
    Random Fourier features with two kernels and mean embedding
    """
    X,y = misr.load_MISR(2)
    index = np.array([x for x in range(len(X))])
    np.random.shuffle(index)
    X_train = [x for (i,x) in enumerate(X) if i in index[:700]]
    X_test = [x for (i,x) in enumerate(X) if i in index[700:]]
    y_train = [Y for (i,Y) in enumerate(y) if i in index[:700]]
    y_test = [Y for (i,Y) in enumerate(y) if i in index[700:]]
    rffe_obj = rffe.randomff_ext(D1 = 100, D2 = 100, p = 16,
                                 lamda = 0.01, sigma1 = 1, sigma2 = 1)
    rffe_obj.train(X_train,y_train)
    print(rffe_obj.test(X_test,y_test))

