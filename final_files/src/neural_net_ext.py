import numpy as np
import random

class Network:
    """
    Specific neural net structure used exclusively for fitting architecture
    with hidden layer - mean pooling - hidden layer - output layer.
    It is only general enough to accept arbitrary loss functions
    and arbitrary activation functions acting in the hidden  layers.
    """
    def __init__(self, layer_sizes, bag_size, loss, activation_fn_1, activation_fn_2):
        """
        Initialize like a regular neural net object, however, the bias
        term is never present beyond the first layer and bag_size
        is the number of observations within each bag
        """
        self.ls = layer_sizes

        # temporary variables which help with creating weight objects
        layer_sizes_weights = np.copy(layer_sizes)
        layer_sizes_weights[1:-1] = 2 * layer_sizes_weights[1:-1]

        # ditto, but with activation objects
        layer_sizes_activ = np.copy(layer_sizes_weights)
        layer_sizes_activ[1] = layer_sizes_activ[1] * bag_size

        # store info for later
        self.loss = loss
        self.a1 = activation_fn_1
        self.a2 = activation_fn_2
        self.bag_size = bag_size

        # initialize biases, weights etc.
        self.bias = np.random.randn(layer_sizes[1])
        self.b_grad = np.zeros(self.bias.shape)

        self.weights = [np.random.randn(i,j) for (i,j)
                        in zip(layer_sizes[1:], layer_sizes_weights[:-1])]
        self.w_grad = [np.zeros(w.shape) for w in self.weights]

        # we del with first layer in a different way, passing all observations
        # from a bag before proceeding to the next layer (mean pooling)
        # that's why we need so many scores in the first layer
        self.scores_l1 = [np.empty(layer_sizes[1]) for i in range(bag_size)]
        self.scores = [np.empty(i) for i in layer_sizes]
        self.deltas = [np.empty(i) for i in layer_sizes]
        self.activations = [np.empty(i) for i in layer_sizes_activ]

        # these are activations we get after mean pooling
        self.pooled_activ = np.empty(2 * layer_sizes[1])

    def feedforward(self,x_batch, light_ver = False):
        """
        Routine for propagating empirical distribution of x
        given by the x_batch through the neural network
        """
        # first layer:
        n = len(x_batch)
        pooled_total = np.zeros(2 * self.ls[1])
        for (i,x) in enumerate(x_batch):
            # scores:
            z = np.dot(self.weights[0],x) + self.bias[1]
            self.scores_l1[i] = z
            # activations:
            a = np.append(self.a1.eval(z,self.ls[1]),
                          self.a2.eval(z,self.ls[1]))
            self.activations[1][2*i*self.ls[1]:
                                2*(i+1)*self.ls[1]] = a
            # summing activations for pooling:
            pooled_total = pooled_total + a
        
        # pooling:
        self.pooled_activ = pooled_total/n

        # second layer:
        # scores:
        self.scores[2] = np.dot(self.weights[1],self.pooled_activ)
        # activations:
        self.activations[2] = np.append(self.a1.eval(self.scores[2],self.ls[2]),
                                        self.a2.eval(self.scores[2],self.ls[2]))

        # output (third) layer
        self.scores[3] = np.dot(self.weights[2], self.activations[2])
        self.activations[3] = self.loss.activ(self.scores[3])
        
        return self.activations[3]

    def backprop(self, x_batch, y):
        """
        Routine for backpropagating information from the loss on the empirical
        distribution of x given by the x_batch propagated through the network
        """
        # delta from the last layer:
        self.deltas[3] = self.loss.delta(y, self.activations[3],self.scores[3])
        # upgrade the gradient for last (third) layer
        self.w_grad[2] = self.w_grad[2] + np.outer(self.deltas[3],
                                                   self.activations[2])

        # delta from the second layer:
        temp = np.dot(self.weights[2].transpose(),self.deltas[3]) * \
               np.append(self.a1.prime(self.scores[2], self.ls[2]),
                         self.a2.prime(self.scores[2], self.ls[2]))
        self.deltas[2] = temp[:self.ls[2]] + temp[self.ls[2]:]

        # weight gradient for the second layer:
        self.w_grad[1] = self.w_grad[1] + np.outer(self.deltas[2],
                                                   self.pooled_activ)

        # delta from the first layer:
        self.w_grad[0].fill(0)
        self.b_grad.fill(0)
        for (x,z) in zip(x_batch,self.scores_l1):
            temp = np.dot(self.weights[1].transpose(), self.deltas[2]) * \
                   np.append(self.a1.prime(z, self.ls[1]),
                             self.a2.prime(z, self.ls[1]))
            delta = temp[:self.ls[1]] + temp[self.ls[1]:]
            # update bias gradient
            self.b_grad = self.b_grad + delta/self.bag_size
            # update weight gradient
            self.w_grad[0] = self.w_grad[0] + np.outer(delta,x)/self.bag_size

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
                if self.loss != Quad_reg:
                    tr = self.test_network(training_data)
                    te = self.test_network(test_data)
                    print("Accuracy on the training dataset: {}.\nLoss on training: {}".format(tr[1], tr[0]))
                    print("Accuracy on the test dataset: {}.\nLoss on testing: {}".format(te[1], te[0]))
                    print("\n------------------------------\n\n\n")
                else:
                    tr = self.test_network(training_data, False)
                    te = self.test_network(test_data, False)
                    print("RMSE on the training dataset: {}.\nLoss on training: {}".format(tr[1], tr[0][0]))
                    print("RMSE on the test dataset: {}.\nLoss on testing: {}".format(te[1], te[0][0]))
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
        # update the weights and biases using
        # the change in biases and weights matrices
        [np.copyto(w,w *(1 - step*lmbd/n) - step/len(batch)*wg)
         for w,wg in zip(self.weights, self.w_grad)]
        self.bias = self.bias -step/len(batch) * self.b_grad

    def test_network(self, test_data, classification = True):
        """
        Propagate data through the network to calculate
        penalized loss and accuracy
        """
        total_loss = 0
        correct = 0
        
        for (x,y) in test_data:
            # prediction
            y_hat = self.feedforward(x)
            # loss on prediction y_hat
            total_loss = total_loss + self.loss.eval(y, y_hat)
            # verify if prediction is correct
            if classification:
                correct = correct + (np.argmax(y_hat) == np.argmax(y))
        # calculate regularized loss
        penalized_loss = total_loss / len(test_data) + \
                         0.5 * self.lmbd * sum(np.linalg.norm(w)**2 for w in self.weights)
        if classification:
            return (penalized_loss, correct/len(test_data))
        else:
            return (penalized_loss, np.sqrt(total_loss/len(test_data)))



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
        return 0.5*(y_hat-y)**2

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
      
