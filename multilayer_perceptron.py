import numpy as np

def SIG(x):
    """ sigmoid function """
    return 1/(1 + np.exp(-x))
def dSIG(x):
    """ derivative of sigmoid """
    return np.exp(-x)/(1 + np.exp(-x))**2
def ReLU(x):
    """ rectifier function """
    result = x*(x > 0)
    return result
def dReLU(x):
    """ derivative of rectifier """
    return 1.*(x>0)
def TANH(x):
    return np.tanh(x)
def dTANH(x):
    return 1 - np.tanh(x)**2
def addOnesCol(X):
    """ add column of ones """
    a, b = np.shape(X)
    Xt = np.zeros([a, b + 1])
    Xt[:,1:] = X
    Xt[:,0] = np.ones(a)
    return Xt

class Mlp():
    """
    Implements multi-layer perceptron 
    """

    def __init__(self, dims=[35, 25, 15, 5, 15, 25, 35], eta=0.001, activation='sigmoid', stochastic=0.,
                 max_epochs=10000, deltaE=-np.inf, alpha=0.8):
        """
        dims = [dim_in, dim_hidden1, ..., dim_hiddenN, dim_out]
        eta = leraning rate 
        activation = activation function
        stochastic = fraction of randomly shuffled training data to use in each epoch. If zero, not stochastic. 
        max_epochs = maximum number of epochs during training
        deltaE = stopping criterion
        alpha = momentum parameter 
        """
        self.set_params(dims, eta, activation, stochastic,
                        max_epochs, deltaE, alpha)

    # compatibility with sklearn
    def set_params(self, dims, eta, activation, stochastic, max_epochs, deltaE, alpha):
        self.dims = dims
        self.eta = eta
        self.activation = activation
        self.stochastic = stochastic
        self.max_epochs = max_epochs
        self.deltaE = deltaE
        self.alpha = alpha
        self.dW = []  # momentum terms
        if activation == 'sigmoid':
            self.f = SIG
            self.df = dSIG
        elif activation == 'linear':
            self.f = lambda x: x
            self.df = lambda x: 1
        elif activation == 'relu':
            self.f = ReLU
            self.df = dReLU
        elif activation == 'tanh':
            self.f = TANH
            self.df = dTANH
        else:
            raise ValueError("invalid activation function %r" % activation)

        return self

    def get_params(self, deep=False):
        result = {
            'dims': self.dims,
            'eta': self.eta,
            'max_epochs': self.max_epochs,
            'activation': self.activation
        }
        return result

    def fit(self, X, y, Xtest=None, ytest=None, weights=None):
        print('X', X[0])
        print('y', y[0])
        if X.shape[0] != y.shape[0]:
            raise ValueError("training and target shapes don't match")
        # initialize weights
        self.weights = weights
        if self.weights is None:
            self.weights = []
            for i in range(len(self.dims)-1):
                W = np.random.rand(self.dims[i+1], self.dims[i] + 1) - 0.5
                #                  ^ output dim    ^ input dim plus bias dim
                # W = (W.T/np.sum(W, axis=1)).T # normalize ROWS for mid-range output
                self.weights.append(W)
        # initial momentum terms
        for W in self.weights:
            self.dW.append(np.zeros(W.shape))
        # store error values
        self.train_error = np.zeros(self.max_epochs+1)
        self.test_error = np.zeros(self.max_epochs+1)
        self.train_error[-1] = np.infty
        self.test_error[-1] = np.infty
        # main training loop
        t = 0
        while (t < self.max_epochs):
            # shuffle data
            Xs = X
            ys = y
            if self.stochastic:
                cut = int(self.stochastic*X.shape[0])
                p = np.random.permutation(X.shape[0])[:cut]
                Xs = X[p, :]
                ys = y[p]
            # forward pass
            Y, x, u = self._forwardpass(Xs, self.weights)
            # compute error
            rmse = self._RMSE(Y, ys)
            self.train_error[t] = rmse
            if Xtest is not None:
                rmse = self.score(Xtest, ytest)
                self.test_error[t] = rmse
                delta = self.test_error[t] - self.test_error[t-1]
            else:
                delta = self.train_error[t] - self.train_error[t-1]
            if abs(delta) < self.deltaE:
                break
            else:
                # backward pass
                self.weights = self._backwardpass(ys, Y, x, u, self.weights)
                t = t + 1
        self.train_error = self.train_error[:t]
        self.test_error = self.test_error[:t]
        # self.weights = weights
        return self.weights

    def predict(self, X):
        Y, _, __ = self._forwardpass(X, self.weights)
        return Y

    def score(self, X, y):
        yp = self.predict(X)
        return self._RMSE(y, yp)

    def _RMSE(self, y, yp):
        """
        Computes the root mean squared error (RMSE)
        """
        return np.sqrt(np.sum((yp-y)**2)/y.shape[0])

    def _forwardpass(self, X, weights):
        """ perform forward pass, saving values"""
        Y = X
        x = []  # inputs to next layer
        u = []  # activations
        for i in range(len(weights)):
            X = addOnesCol(Y)
            x.append(X)             # save input
            U = (weights[i]@X.T).T  # apply weight matrix
            u.append(U)             # save output
            Y = self.f(U)           # activated output
        return Y, x, u

    def _backwardpass(self, y, Y, x, u, weights):
        """ 
        Compute updated weights by doing backward pass
        y = target 
        Y = true output 
        x = inputs to weight matrices at each layer during forward pass
        u = activations at each output layer during forward pass 
        """
        # backward pass
        D = -self.df(u[-1])*(y - Y)  # Delta
        delta = [D]
        for i in range(len(weights) - 1):
            W = weights[::-1][i]   # go through weight matrices in reverse
            U = u[::-1][i+1]  # go through outputs in reverse, from second last
            d = self.df(U)*(delta[i]@W)[:, 1:]
            delta.append(d)
        delta.reverse()  # reverse delta!
        # update weights
        weights_new = []
        for i in range(len(weights)):
            W = weights[i]
            momentum = self.alpha*self.dW[i]
            learningTerm = self.eta*(delta[i].T @ x[i])
            Wnew = W - learningTerm + momentum
            self.dW[i] = Wnew - W
            weights_new.append(Wnew)
        return weights_new

    def __repr__(self):
        return "in:%i, hidden:%i out:%i " % self.dims
