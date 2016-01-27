import numpy as np
import random

class Network:
    """
    Class which holds scaffolding of an artificial neural network,
    and holds methods for training, testing and prediction.
    The layer sizes are specified using one of the parameters.
    The 0th layer corresponds to the dimension of the input data.
    The last layer corresponds to the number of classes or set to 1 if regression.
    """
    def __init__(self, layer_sizes, loss, activation_fn_1, activation_fn_2, include_bias = True):
        """
        Initialize bias, weight, gradient, weight and bias change,
        score, activation and delta matrices for each layer.
        Store some additional variables.
        """
        # variable manipulation
        layer_sizes = np.array(layer_sizes)
        aux_layer_sizes = np.copy(layer_sizes)
        aux_layer_sizes[1:-1] = 2 * layer_sizes[1:-1]
        self.incl_b = include_bias

        # store important variables
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.loss = loss
        self.ms = np.copy(layer_sizes[1:-1])
        
        # momentum seems to work poorly when SGD is in use
        # so by default it is disabled
        self.moment = 0
        self.a1 = activation_fn_1
        self.a2 = activation_fn_2

        # pre-allocate memory for all the matrices
        # this should speed up training
        self.biases = [np.random.randn(i) for i in layer_sizes]
        if not include_bias:
            [b.fill(0) for b in self.biases[2:]]
        self.b_grad = [np.zeros(b.shape) for b in self.biases]
        self.b_change = [np.zeros(b.shape) for b in self.biases]
        
        self.weights = [np.random.randn(i,j) for (i,j) in
                        zip(layer_sizes[1:], aux_layer_sizes[:-1])]
        self.w_grad = [np.zeros(w.shape) for w in self.weights]
        self.w_change = [np.zeros(w.shape) for w in self.weights]
        
        self.scores = [np.empty(i) for i in layer_sizes]
        self.deltas = [np.empty(i) for i in layer_sizes]
        self.activations = [np.empty(i) for i in aux_layer_sizes]

    def feedforward_light(self, x):
        """
        Propagates input x through the Neural Network
        in order to obtain the predicted value
        """
        # propagate through layers 1,...,L-1
        for (i,(b,w)) in enumerate(zip(self.biases[1:-1],self.weights[:-1])):
            m = self.ms[i]
            z = np.dot(w,x) + b
            x = np.append(self.a1.eval(z,m), self.a2.eval(z,m))

        # propagate through the last layer, and return predicted value
        z = np.dot(self.weights[-1],x) + self.biases[-1]
        x = self.loss.activ(z)
        return x

    def feedforward(self,x):
        """
        Propagates input x through the Neural Network
        in order to obtain all the activations and scores
        """
        # calculate scores for the first layer
        self.activations[0] = x
        self.scores[1] = np.dot(self.weights[0],x)+ self.biases[1]
        
        # propagate through layers 1,...,L-1, and calculate
        # the scores for the last layer
        for i in range(1, self.num_layers - 1):
            m = self.ms[i-1]
            self.activations[i] = np.append(self.a1.eval(self.scores[i], m),
                                            self.a2.eval(self.scores[i], m))
            self.scores[i+1] = np.dot(self.weights[i],self.activations[i]) + \
                               self.biases[i+1]
        # propagate through last layer
        self.activations[-1] = self.loss.activ(self.scores[-1])

    def backprop(self,x,y):
        """
        Backpropagates the information from the loss function J(f(x),y)
        to weights and biases through gradients
        """
        # update gradients from last layer
        self.deltas[-1] = self.loss.delta(y, self.activations[-1],
                                          self.scores[-1])
        self.w_grad[-1] = self.w_grad[-1] + np.outer(self.deltas[-1],
                                                     self.activations[-2])
        if self.incl_b:
            self.b_grad[-1] = self.b_grad[-1] + self.deltas[-1]

        # update gradients of all hidden layers 
        for l in reversed(range(1,self.num_layers-1)):
            m = self.ms[l-1]
            mid = self.layer_sizes[l]

            # since weights are shared between units i and (i+m)
            # for i =0,...,(m-1), delta is formed by this "midpoint" method
            temp = np.dot(self.weights[l].transpose(),self.deltas[l+1]) * \
                             np.append(self.a1.prime(self.scores[l], m),
                                       self.a2.prime(self.scores[l], m))
            self.deltas[l] = temp[:mid] + temp[mid:]

            # weight gradient
            self.w_grad[l-1] = self.w_grad[l-1] + \
                               np.outer(self.deltas[l],
                                        self.activations[l-1])
            # bias gradient if appropriate
            if l == 1 or self.incl_b:
                self.b_grad[l] = self.b_grad[l] + self.deltas[l]

    def train_network(self, training_data, iterations, batch_size, step,
            lmbd = 0.0, test_data = None):
        """
        Trains the neural network using Stochastic Gradient Descent,
        with batch updates - i.e. instead of using estimate of grad_E(J)
        using all data points it uses some small batch of data instead.
        """
        # store variables
        n = len(training_data)
        self.lmbd = lmbd
        
        # loop over iterations
        for j in range(iterations):
            print("\n-----------------------------\n")
            print("starting iteration {}".format(j))

            # Split the randomly shuffled data into batches
            random.shuffle(training_data)
            batches = [training_data[k:k+batch_size]
                       for k in range(0, n, batch_size)]

            # for each batch perform an update step
            for batch in batches:
                self.batch_update(batch, n, step, lmbd)

            # print out info about accuracy if test data is present
            if test_data:
                tr = self.test_network(training_data)
                te = self.test_network(test_data)
                print("Accuracy on the training dataset: {}.\nLoss on training: {}".format(tr[1], tr[0][0]))
                print("Accuracy on the test dataset: {}.\nLoss on testing: {}".format(te[1], te[0][0]))
                print("\n------------------------------\n\n\n")
                
    
    def batch_update(self, batch, n, step, lmbd):
        """
        Use samples from the batch to compute mean gradients
        and update the weights and biases
        """
        # clean up matrices
        [w.fill(0) for w in self.w_grad]
        [b.fill(0) for b in self.b_grad]

        # propagate forwards and backwards all data in batch
        for x, y in batch:
            self.feedforward(x)
            self.backprop(x, y)
        # update the change in biases and weights matrices
        # (these are used for momentum mainly)
        [np.copyto(wch,step*lmbd/n*w + step/len(batch)*nw + self.moment * wch)
                        for w, nw, wch in zip(self.weights, self.w_grad, self.w_change)]
        [np.copyto(bch,step/len(batch)*nb + self.moment * bch)
                         for b,nb, bch in zip(self.biases, self.b_grad, self.b_change)]
        # update the weights and biases using
        # the change in biases and weights matrices
        [np.copyto(w,w - wch) for w,wch in zip(self.weights, self.w_change)]
        [np.copyto(b,b - bch) for b,bch in zip(self.biases, self.b_change)]
    
    def test_network(self, test_data):
        """
        Propagate data through the network to calculate
        penalized loss and accuracy
        """
        total_loss = 0
        correct = 0.0
        for (x,y) in test_data:
            # prediction
            y_hat = self.feedforward_light(x)
            # loss on prediction y_hat
            total_loss = total_loss + self.loss.eval(y, y_hat)
            # verify if prediction is correct
            correct = correct + 1.0*(np.argmax(y_hat) == np.argmax(y))
        # calculate regularized loss
        penalized_loss = total_loss / len(test_data) + \
                         0.5 * self.lmbd * sum(np.linalg.norm(w)**2 for w in self.weights)
        return (penalized_loss, correct/len(test_data))


class Entropy:
    """
    Class for loss function using the log likelihood loss
    and softmax in the last activation layer
    """
    @staticmethod
    def delta(y,  y_hat, z = None):
        return y_hat - y

    @staticmethod
    def eval(y, y_hat):
        y_index = np.where(np.array(y))
        return -np.nan_to_num(np.log(y_hat[y_index]))

    @staticmethod
    def activ(y):
        m = np.amax(y)
        return np.exp(y - m)/np.sum(np.exp(y - m))
    
class Quad:
    """
    Class for squared error loss function using
    sigmoids in the last activation layer
    """
    @staticmethod
    def delta(y, y_hat, z):
        return (y_hat-y) * sigmoid_prime(z)

    @staticmethod
    def eval(y, y_hat):
        return 0.5*np.linalg.norm(y_hat-y)**2

    @staticmethod
    def activ(y):
        return sigmoid(y)

class Quad_reg:
    """
    Class for squared error loss function using 
    identities in the last activation layer
    """
    @staticmethod
    def delta(y, y_hat, z):
        return (y_hat-y)

    @staticmethod
    def eval(y, y_hat):
        return 0.5*np.linalg.norm(y_hat-y)**2

    @staticmethod
    def activ(y):
        return y

class Sigmoid:
    """
    Class for sigmoid activation function used by
    the units in the neural nets
    """
    @staticmethod
    def eval(y, m = None):
        return sigmoid(y)

    @staticmethod
    def prime(y, m = None):
        return sigmoid_prime(y)

class Cos:
    """
    Class for cosine activation function used by
    the units in the neural nets
    """

    @staticmethod
    def eval(y, m):
        return np.cos(y) / np.sqrt(m)

    @staticmethod
    def prime(y, m):
        return - np.sin(y) / np.sqrt(m)

class Sin:
    """
    Class for sine activation function used by
    the units in the neural nets
    """

    @staticmethod
    def eval(y, m):
        return np.sin(y) / np.sqrt(m)

    @staticmethod
    def prime(y, m):
        return np.cos(y) / np.sqrt(m)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
