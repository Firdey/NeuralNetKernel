import numpy as np
class Network:
        def __ init__(self, layer_size, loss, activ, xdata, ydata, epsilon, alpha, penalty, max_it):
# layer_size is a 3 dimensional vector with [input size m (with cos and sin each) output size]

# loss is an object being pass defining the loss function with also penalty, current choices are
#  quadratic loss, hinge loss logistic loss

# activ refers to the activation function used in final layer, e.g. soft max

# Xdata is the X observatios dimension n by p

# Ydata is the Y observations dimension n

                self.layer_size = layer_size
                self.loss = loss
                self.activ = activ
                self.xdata = xdata
                self.ydata = ydata
                self.n = len( ydata )
                self.p = layer_size[ 0 ]
                self.m = layer_size[ 1 ]
                self.weights = np.ones( ( self.m , self.p ) ) * 0.05
                self.betas = np.ones( 2 * self.m ) * 0.05
                self.max_it = max_it
                self.iterations = 0
                self.loss_vector = np.zeros( max_it )
                self.lam = penalty[0]
                self.mu = penalty[1]
                self.epsilon = epsilon
                self.alpha = alpha
                self.grad_w_loss = np.array( [ [ 0 for i in range( 0, self.p ) ] for j in range( 0, self.m ) ] )
                self.step_method = step_method
                self.weight_diff = 0
                self.betas_diff = 0

        # forward propagation
        #compute fitted values etc
        def forward( self ):
                self.xdata_weight = np.dot( self.weights, self.xdata.transpose( ) )
                self.hidden_matrix_sin = np.array( sin( self.xdata_weight ) * ( 1.0 / ( np.sqrt( self.m ) )
                self.hidden_matrix_cos = np.array( cos( self.xdata_weight ) * ( 1.0 / ( np.sqrt( self.m ) )
                self.hidden_matrix = np.array( [ [ self.hidden_matrix_cos ], [ self.hidden_matrix_sin ] ] )
                self.weight_matrix = np.dot( self.betas, self.hidden_matrix )
                self.ydata_hat = activ.formula( self.weight_matrix )


        #backward propagation
        #updates weights and betas
        def backward ( self ):
                y_errors = ydata - ydata_hat
                self.loss_vector[ self.iterations ] =  np.dot( y_errors, y_errors ) * 1.0 / ( self.n ) + self.lam * np.dot( self.betas, self.betas ) + self.mu * np.sum( self.weights * self.weights )
                s_grad = activ.diff( self.weight_matrix )
                grad_b_loss = 2 * self.lam * self.betas - np.dot( (2 / self.n) * s_grad * self.hidden_matrix , y_errors )
                self.betas_diff = (self.epsilon * grad_b_loss + self.alpha * self.betas_diff)
                self.betas = self.betas - betas_diff
                betas_cos = self.betas[ 0 : self.m ]
                betas_sin = self.betas[ self.m : 2 * self.m ]

                adjust_hidden_matrix = ( self.hidden_matrix * s_grad )

                for j in range( 0, self.m ):
                        term_1 = np.dot( y_errors, self.xdata * self.hidden_matrix_cos[ :, j ] * s_grad * betas_sin[ j ] )
                        term_2 = np.dot( y_errors, self.xdata * self.hidden_matrix_sin[ :, j ] * s_grad * betas_cos[ j ] )
                    self.grad_w_loss[ j ] = 2 * self.lam * self.weights + ( 2.0/self.n ) * ( term_1 - term_2 )
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
 

Sigmoid.prime(np.array([1,2,3]))
