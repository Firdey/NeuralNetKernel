import numpy as np
class Network:
    def __init__(self, layer_size, loss, activ, xdata, ydata, epsilon, alpha, penalty, max_it, tolerance):

        # layer_size is a 3d with [ input size, hidden layer size and output size ] 

        # loss is an object with a method which evaluates the loss formula (e.g. quadratic, hinge etc)

        # activation function is an object

        # activation function is an object which has a method evaluating the activation function

        # xdata is a numpy array of the X observations dim = n x p

        # xdata is a numpy array of the Y observations dim = n 

        # epsilon is the step size in the propagation algo

        # alpha is the momentum coefficient

        # penalty is the lambda and mu 

        # max_it is the maximum number of iterations before we stop the algo

        # tolerance is another stopping criterion for the errors


        self.layer_size = layer_size # layer_size is a 3d with [ input size, hidden layer size and output size ] 
        self.loss = loss 			 # loss is an object with a method which evaluates the loss formula (e.g. quadratic, hinge etc)
        self.activ = activ 			 # activation function is an object which has a method evaluating the activation function
        self.xdata = xdata_weight 	 # xdata is a numpy array of the X observations dim = n x p
        self.ydata = ydata           # xdata is a numpy array of the Y observations dim = n 
        self.n = len( ydata )		 #	
        self.p = layer_size[ 0 ]
        self.m = layer_size[ 1 ]
        self.weights = np.ones( ( self.m , self.p ) ) * 0.05 # CHANGE
        self.betas = np.ones( 2 * self.m ) * 0.05 #Randomised function np.rand.randn
        self.max_it = max_it
		self.lam = penalty[ 0 ]
        self.mu = penalty[ 1 ]
        self.epsilon = epsilon
        self.alpha = alpha
        self.tolerance = tolerance

        # some more initialisations  
        self.iterations = 0  # set number of iter = 0
        self.loss_vector = np.zeros( max_it ) # the errors for the graph
        self.grad_w_loss = np.zeros ( ( self.p , self.m )  ) #
        self.weight_diff = 0
        self.betas_diff = 0
    

        # forward propagation
        #compute fitted values etc
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
        grad_b_loss = 2 * self.lam * self.betas - np.dot( s_grad * self.hidden_matrix , self.loss.diff(self.ydata, self.ydata_hat) )
        self.betas_diff = (self.epsilon * grad_b_loss + self.alpha * self.betas_diff)
        self.betas = self.betas - betas_diff
        betas_cos = self.betas[ 0 : self.m ]
        betas_sin = self.betas[ self.m : 2 * self.m ]

        for j in range( 0, self.m ):
            term_1 = np.dot( self.loss.diff(self.ydata, self.ydata_hat), self.xdata * self.hidden_matrix_cos[ :, j ] * s_grad * betas_sin[ j ] )
            term_2 = np.dot( self.loss.diff(self.ydata, self.ydata_hat), self.xdata * self.hidden_matrix_sin[ :, j ] * s_grad * betas_cos[ j ] )
            self.grad_w_loss[ j ] = 2 * self.lam * self.weights + ( term_1 - term_2 )
            self.weight_diff = (self.epsilon * self.grad_w_loss + self.alpha * self.weight_diff)
            self.weights = self.weights - self.weight_diff  #might need to transpose
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

