import numpy as np

'''
function to compute the weights beta when the angular frequencies w are simulated
from a gaussian kernel 
the function accepts the 
data x as a n by p array
reponses y as an n dimensional array
size of the hiddel layer (ie the number of nodes)
penalty coefficient lamda for ridge regression
'''
class randomff:
    def __init__(self, D, p, lamda = 0, sigma = 1):
        self.D = D
        self.lamda = lamda
        self.sigma = sigma
        mean = np.zeros(p) #mean for the w
        cov = sigma * np.identity(p) # covariance of the angular frequencies w
        self.w = np.random.multivariate_normal(mean, cov, D)
        self.bias = np.random.uniform(0,2*np.pi,D)

    def train(self, xdata, ydata):
        print("\nStarting to train the model.\n")
        lamda = self.lamda
        D = self.D
        z = self.embed(xdata)
        self.beta_hat = np.linalg.solve(np.dot(z.T,z) + lamda * np.identity( 2*D ),
                                        np.dot( z.T, ydata))
        print("\nThe model has been trained successfully!\n")

    def predict(self, xdata):
        z = self.embed(xdata)
        y_hat = np.dot(z, self.beta_hat)
        return y_hat

    def embed(self,xdata):
        n = xdata.shape[0]
        w = self.w
        D = self.D
        biases = np.array([self.bias for i in range(n)])
        z = np.dot(xdata, w.T) + biases
        z_cos = np.cos(z) / np.sqrt(D)
        z_sin = np.sin(z) / np.sqrt(D)
        z = np.hstack( ( z_cos, z_sin ) )
        return z


    def test(self, xdata, ydata):
        y_hat = self.predict(xdata)
        misclass = abs((y_hat>0.5)-ydata)
        return sum(misclass)/len(ydata)


