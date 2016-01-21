import numpy as np

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
        #   loss: object with a method which evaluates the loss formula (e.g. quadratic, hinge etc) and its gradient
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
        self.weights = np.ones( ( self.m , self.p ) ) * 0.05 #matrix of angular frequncies (m x p matrix)
        self.betas = np.ones( 2 * self.m ) * 0.05 #vector of 2m betas (amplitudes)
        self.loss_vector = np.zeros( max_it ) #the loss at each step of backward propagation
        self.iterations = 0  #number of iterations done so far

        self.weight_diff = 0 #THIS SHOULD BE IN BACKWARD METHOD
        self.betas_diff = 0 #THIS SHOULD BE IN BACKWARD METHOD
        self.grad_w_loss = np.zeros ((self.p , self.m )) #THIS SHOULD BE IN BACKWARD METHOD
    

    #METHODS

    #FORWARD PROPAGATION
        #given the parameters predict the response vectors
    def forward( self ):
        self.xdata_weight = np.dot( self.weights, self.xdata.transpose( ) ) #this is  dim m X p of weighted data
        self.phi_sin = np.array( np.sin( self.xdata_weight ))  / np.sqrt( self.m ) # sin part of phi
        self.phi_cos = np.array( np.cos( self.xdata_weight ))   / np.sqrt( self.m ) # cos part of phi 
        self.phi_matrix = np.vstack( (  self.phi_cos , self.phi_sin ) ) # stuck cosines on top of sines
        self.weight_vector = np.dot( self.betas, self.phi_matrix ) # dim n  includes paramters inside the activation function
        self.ydata_hat = self.activ.formula( self.weight_vector) # the fitted y's that depend on the activation fn


        #backward propagation
        #updates weights and betas
    def backward ( self ):
        self.loss_vector[ self.iterations ] =  self.loss.formula(self.ydata, self.ydata_hat) * 1.0 / ( self.n ) + self.lam * np.dot( self.betas, self.betas ) + self.mu * np.sum( self.weights * self.weights )
        #loss vector is the evaluation of our objective function in every iteration
        s_grad = self.activ.diff( self.weight_matrix )
        #s_grad computes the gradient of the activation function
        grad_b_loss = 2 * self.lam * self.betas - np.dot( s_grad * self.phi_matrix , self.loss.diff(self.ydata, self.ydata_hat) ) 
        #gradient of the objective function wrt b
        self.betas_diff = (self.epsilon * grad_b_loss + self.alpha * self.betas_diff)
        #this is the betas difference including the step*grad and for the momentum term
        self.betas = self.betas - self.betas_diff
        betas_cos = self.betas[ 0 : self.m ]
        betas_sin = self.betas[ self.m : 2 * self.m ]

        for j in range( 0, self.m ):
            help_mat =  - self.loss.diff(self.ydata, self.ydata_hat) * s_grad * ( self.phi_cos[ j, : ] *  betas_sin[ j ] - self.phi_sin[ j, : ] *  betas_cos[ j ])* self.xdat.transpose() 
            #this gives a p by n matrix where each data point i is scaled by the appropiate coef 
            term_1 = np.array( [sum(column) for column in help_mat]).transpose()
            #now we sum the columns of the matrix to obtain a p vector for the gradient wrt w_r
            self.grad_w_loss[ j ] = 2 * self.lam * self.weights + term_1 
            self.weight_diff = (self.epsilon * self.grad_w_loss + self.alpha * self.weight_diff)
            self.weights = self.weights - self.weight_diff  
            self.iterations = self.iterations + 1

    def train ( self ):
        for h in range( 0, max_it):
            self.forward()
            self.backward()

class Sigmoid:
        @staticmethod
        def formula(z):
            return 1/(1+np.exp(-z))
        @staticmethod
        def diff(z):
            return Sigmoid.value(z) * (1-Sigmoid.value(z))
class Softmax:
    @staticmethod
    def formula(z):
        return np.exp(z)/np.sum(np.exp(z))
    @staticmethod
    def diff(z):
        return
class Mean_S_E:
    @staticmethod
    def formula(ydata, ydata_hat):
        y_errors = ydata - ydata_hat
        return np.dot( y_errors, y_errors ) * 1.0 / ( self.n )
    @staticmethod
    def diff(ydata, ydata_hat):
        y_errors = ydata - ydata_hat
        return 2 * y_errors * 1.0 / ( self.n )

