import os, struct
from array import array as pyarray
import numpy as np
from numpy import append, array, int8, uint8, zeros

def load_mnist(dataset="training", digits=np.arange(10), path="./data/"):
    """
    This function is taken from the internet, I did not write it! Marcin.

    Loads MNIST files into 3D numpy arrays
    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'MNIST/train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'MNIST/train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 'MNIST/t10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'MNIST/t10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=float)
    labels = zeros((N, 1), dtype=float)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels

def load_mnist_wrapper2(path = "./data/"):
    train = load_mnist("training", path = path)
    train_x = [x.reshape(784)/255 for x in train[0]]
    train_y = [np.zeros(10) for y in train[1]]
    [y.itemset(i,1) for (y,i) in zip(train_y, train[1])]
    test = load_mnist("testing", path = path)
    test_x = [x.reshape(784)/255 for x in test[0]]
    test_y = [np.zeros(10) for y in test[1]]
    [y.itemset(i,1) for (y,i) in zip(test_y, test[1])]
    return ([(x,y) for x,y, in zip(train_x, train_y)],
            [(x,y) for x,y in zip(test_x, test_y)])

def load_mnist_wrapper(path = "./data/"):
    train = load_mnist("training", path = path)
    train_x = [x.reshape(784,1)/255 for x in train[0]]
    train_y = [np.zeros([10,1]) for y in train[1]]
    [y.itemset(i,1) for (y,i) in zip(train_y, train[1])]
    test = load_mnist("testing", path = path)
    test_x = [x.reshape(784,1)/255 for x in test[0]]
    test_y = [np.zeros([10,1]) for y in test[1]]
    [y.itemset(i,1) for (y,i) in zip(test_y, test[1])]
    return ([(x,y) for x,y, in zip(train_x, train_y)],
            [(x,y) for x,y in zip(test_x, test_y)])
