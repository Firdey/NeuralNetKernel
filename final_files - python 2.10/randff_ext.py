import numpy as np

class randomff_ext:
    def __init__(self, D1, D2, p, lamda = 0, sigma1 = 1, sigma2 = 1):
        self.D1 = D1
        self.D2 = D2
        self.lamda = lamda

        mean = np.zeros(p)
        cov = sigma1 * np.identity(p)
        self.w1 = np.random.multivariate_normal(mean, cov, D1)
        self.bias1 = np.random.uniform(0,2*np.pi,D1)

        mean = np.zeros(2*D1)
        cov = sigma2 * np.identity(2*D1)
        self.w2 = np.random.multivariate_normal(mean,cov,D2)
        self.bias2 = np.random.uniform(0,2*np.pi,D2)
        
    def train(self, xdata, ydata):
        print("\nStarting to train the model.\n")
        lamda = self.lamda
        z = self.embed_all(xdata)
        self.beta_hat = np.linalg.solve(np.dot(z.T,z) + lamda * np.identity( 2*self.D2 ),
                                        np.dot( z.T, ydata))
        print("\nThe model has been trained successfully!\n")

    def predict(self, xdata):
        z = self.embed_all(xdata)
        y_hat = np.dot(z, self.beta_hat)
        return y_hat

    def embed(self, xdata, layer = 1):
        n = xdata.shape[0]
        if layer == 1:
            w = self.w1
            D = self.D1
            b = self.bias1
        else:
            w = self.w2
            D = self.D2
            b = self.bias2
        biases = np.array([b for i in range(n)])
        z = np.dot( xdata, w.T ) + biases
        z_cos = np.cos( z ) / np.sqrt( D )
        z_sin = np.sin( z ) / np.sqrt( D )
        z = np.hstack( ( z_cos, z_sin ) )
        return z

    def embed_all(self, xdata):
        mu_x = np.zeros(self.D1 * 2)
        for X in xdata:
            z = self.embed(X,layer = 1)
            mean_embedding = np.mean(z, axis=0)
            mu_x = np.vstack((mu_x, mean_embedding))
        mu_x = mu_x[1:,]
        z = self.embed(mu_x, layer = 2)
        return z

    def test(self, xdata, ydata):
        y_hat = self.predict(xdata)
        RMSE = np.linalg.norm(y_hat-ydata)/np.sqrt(len(y_hat))
        return RMSE
