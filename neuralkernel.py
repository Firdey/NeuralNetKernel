import numpy as np
import matplotlib.pyplot as plt
import math
import csv

#CLASS NETWORK: Neural network using Kernel Learning via Random Fourier Representations
#METHODS:
# FORWARD: Forward propagation, given the parameters predict the response vectors
# BACKWARD: Backward propagation, given the response vectors update the parameters
# TRAIN: Given an inital state, iterate forward backward iteration
class Network:

    #MEMBER VARIABLES: SEE CONSTRUCTOR with the self. prefix

    #CONSTRUCTOR
    def __init__(self, layer_size, loss, activ, xdata, ydata, epsilon, alpha, penalty, max_it, tolerance):

        # PARAMETERS:
        #   layer_size: an array with 3 elements [number of features in X, hidden layer size (m), output size (1 for regresion)]
        #   loss: object with a method which evaluates the loss formula (e.g. MEAN quadratic, hinge etc) and its gradient
        #   activ: activation function is an object with a method which evaluates the activation function and its gradient
        #   xdata: numpy array of the X observations, dimensions = n x p (matrix)
        #   ydata: numpy array of the Y observations, dim = n (vector)
        #   epsilon: step size for gradient descent
        #   alpha: parameter for momentum
        #   penalty: array of size 2 for penalty terms [lambda, mu]
        #   max_it: maximum number of iterations
        #   tolerance: stopping criterion if the difference in loss between steps is less than tolerance

        #ASSIGN CONSTRUCTOR PARAMETERS TO MEMBER VARIABLES
        self.loss = loss #loss object
        self.activ = activ #activ object
        self.xdata = xdata #design matrix
        self.ydata = ydata #response vector
        self.epsilon = epsilon #step size
        self.alpha = alpha #momentum parameter
        self.lam = penalty[0] #penalty term (for beta)
        self.mu = penalty[1] #penalty term (for the angular frequencies)
        self.max_it = max_it #maximum number of iterations
        self.tolerance = tolerance #stopping criterion

        #EXTRACT THE DIMENSIONS of X, y and check if they are consistent
        #X is a nxp matrix
        #Y is a n size vector
        #layer size is a vector of size 3 containing [p,m,k]
        self.n = len(ydata)
        self.p = layer_size[0]
        #ensure n is consistent, if not throw exception
        if (self.n != xdata.shape[0]):
          raise Exception('The height of xdata is not the same as the length of ydata')
        #ensure p is consistent, if not throw exception
        if (self.p != xdata.shape[1]):
          raise Exception('The width of xdata is not the same as layer_size[0]')

        #ADDITIONAL MEMBER VARIABLES
        self.m = layer_size[1] #size of hidden layer
        self.weights = np.random.normal(0.0, 10, (self.m,self.p)) #matrix of angular frequncies (m x p matrix)
        self.betas = np.random.normal(0.0, 10, 2*self.m) #vector of 2m betas (amplitudes)
        self.loss_vector = np.zeros(max_it) #the loss at each step of backward propagation
        self.iterations = 0  #number of iterations done so far

        self.weight_diff = np.zeros((self.m,self.p)) #for momentum
        self.betas_diff = np.zeros(2*self.m) #for momentum

        self.phi_matrix = np.zeros((2*self.m, self.n))
        self.ydata_hat = np.zeros(self.n)
        self.weight_vector = np.zeros( ( self.m , self.p ) )

    #METHODS

    #METHOD: FORWARD PROPAGATION
        #given the parameters predict the response vectors
    def forward( self ):
        xdata_weight = np.dot(self.weights, self.xdata.transpose()) #this is  dim m X p of weighted data
        phi_sin = np.sin(xdata_weight) # sin part of phi
        phi_cos = np.cos(xdata_weight) # cos part of phi
        self.phi_matrix = np.vstack((phi_cos, phi_sin)) / np.sqrt(self.m) # stack cosines on top of sines
        self.weight_vector = np.dot(self.betas, self.phi_matrix) # dim n  includes paramters inside the activation function
        self.ydata_hat = self.activ.formula(self.weight_vector) # the fitted y's that depend on the activation fn


    #METHOD: BACKWARD PROPAGATION
        #given the response vectors, update weights and betas
    def backward ( self ):
        #loss vector is the evaluation of our objective function in every iteration
        self.loss_vector[ self.iterations ] =  self.loss.formula(self.ydata, self.ydata_hat, self.n) * 1.0  + self.lam * np.dot( self.betas, self.betas ) + self.mu * np.sum( self.weights * self.weights )
        #s_grad computes the gradient of the activation function
        s_grad = self.activ.diff(self.weight_vector)
        loss_grad = self.loss.diff(self.ydata, self.ydata_hat, self.n)

        #LAYER 1 - UPDATE BETA

        #gradient of the objective function wrt b
        grad_b_loss = 2 * self.lam * self.betas + np.dot(self.phi_matrix, s_grad * loss_grad)
        #this is the betas difference including the step*grad and for the momentum term
        self.betas_diff = self.epsilon * grad_b_loss + self.alpha * self.betas_diff
        #update the beta parameter
        self.betas -= self.betas_diff

        #LAYER 2 - UPDATE WEIGHTS

        #set the step size due to momentum
        self.weight_diff *= self.alpha #momentum term

        #for each column in the weight matrix, update the weights using gradient descent
        for j in range(0, self.m):

            #gradient of objective wrt weight vector = X' * (diff_loss .* diff_activation .* diff_fourier)
            #where diff_four is a n vector with elements (beta_sin[j] * cos(weight*x) - beta_cos[j] * sin(weight*x))

            grad_w_loss = self.betas[j+self.m]*self.phi_matrix[j,:] - self.betas[j]*self.phi_matrix[j+self.m,:]
            grad_w_loss *= s_grad
            grad_w_loss *= loss_grad
            grad_w_loss = np.dot(self.xdata.transpose(),grad_w_loss)
            grad_w_loss += 2*self.lam*self.weights[j, :]

            #CODE FROM XENIA
            #now we sum the columns of the matrix to obtain a p vector for the gradient wrt w_r
            #term_1 = np.array( [sum(column) for column in help_mat]).transpose()

            #append most recent gradient
            self.weight_diff[j, :] += self.epsilon * grad_w_loss

        #end j for

        #take the step and increment the number of iterations
        self.weights -= self.weight_diff
        self.iterations += 1

    #end def backward(self)

    #METHOD: TRAIN NEURAL NETWORK
      #iterate between forward and backward pass
    def train (self):
        while (self.iterations < self.max_it):
            self.forward()
            self.backward()
            if self.loss_vector[self.iterations-1] < self.tolerance:
              break

#ACTIVATION STATIC CLASSES
class Sigmoid:
    @staticmethod
    def formula(z):
        return 1/(1+np.exp(-z))
    @staticmethod
    def diff(z):
        return Sigmoid.formula(z) * (1-Sigmoid.formula(z))
class Softmax:
    @staticmethod
    def formula(z):
        return np.exp(z)/np.sum(np.exp(z))
    @staticmethod
    def diff(z):
        return
class Linear:
    @staticmethod
    def formula(z):
        return z
    @staticmethod
    def diff(z):
        return np.ones(len(z))

#LOSS STATIC CLASSES
class Mean_S_E:
    @staticmethod
    def formula(ydata, ydata_hat, n):
        y_errors = ydata - ydata_hat
        return np.dot( y_errors, y_errors ) * 1.0 / ( n )
    @staticmethod
    def diff(ydata, ydata_hat, n):
        y_errors = ydata - ydata_hat
        return -2 * y_errors * 1.0 / ( n )


'''
print "Importing data..."
with open('forestfiretrans.csv', 'rb') as csvfile:
    next(csvfile, None)
    forest = csv.reader(csvfile)
    forest_data = [map(float, l) for l in forest]
'''

#IMPORT DATA
print "Importing data..."
with open('SAheart.csv', 'rb') as csvfile:
    next(csvfile, None)
    data_matrix = csv.reader(csvfile)
    data_matrix = [map(float, l) for l in data_matrix]

#GET DIMENSION OF DATA AND PRINT IT
n = len(data_matrix)
p = len(data_matrix[0])-1
print "Data set has "+str(n)+" samples and "+str(p)+" features"

#EXTRACT THE RESPONSE VARAIBLE FROM DATA MATRIX AND REMOVE IT
#ADD CONSTANT (BIAS) TERM
y_response = [0]*n #y_response is a n vector
#for each data
for i in range(0, n):
    #change the row name into a constant
    data_matrix[i][0]=1.0
    #extract the ith response and put it in the y_response vector
    y_response[i] = data_matrix[i][p]
    #remove the response from the data matrix
    #pop(p) = remove the pth element
    data_matrix[i].pop(p)

#DECLARE VARIABLES/OBJECTS FOR NEURAL NETWORK
layer_size = [p, 100, 1] #size 3 vector [number of features, number of hidden nodes, dimension of response variable]
loss = Mean_S_E #static class with a method for the loss and its differential
activ = Sigmoid #static class with a method for the activation function and its differential
X = np.array(data_matrix) #design matrix as a numpy array
y = np.array(y_response) #response vector as a numpy array
epsilon = 0.0001 #step size
alpha = 0.0 #coefficient for momentum
penalty = [0.0, 0.0] #size 2 vector for ridge regression tuning parameters for [betas (amplitudes), weights (angular frequencies)]
max_it = 1000 #number of iterations
tolerance = 0.0 #tolerence

#TRAIN THE NEURAL NETWORK
print "Training classifier..."
neural_network = Network(layer_size, loss, activ, X, y, epsilon, alpha, penalty, max_it, tolerance)
neural_network.train()

#PRINT THE RESULTS
print "Below are the feature vectors, responses and its prediction for each data"
for i in range(0,n):
    print "Feature"+str(data_matrix[i])+" Response: "+str(y[i])+" Prediction: "+str(neural_network.ydata_hat[i])
error = np.sum(np.absolute(y-neural_network.ydata_hat))*100/n
print "Test error: "+str(error)
plt.plot(range(max_it), neural_network.loss_vector)
plt.show()
