import numpy as np

class AdalineSGD:
    def __init__(self, eta=0.01, n_iter=50, shuffle=True, randome_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.random_state = randome_state
        self.initialized_weights = False
    def fit(self, X, y):
        self.initialized_weights(X.shape[1])
        self.losses_ = []
        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            losses = []
            for xi, target in zip(X,y):
                losses.append(self._update_weights(xi, target))
            avg_losses = np.mean(losses)
            self.losses_.append(avg_losses)
        return self   
    def _initialize_weights(self, m):
        self.rgen_ = np.random.RandomState(self.random_state)
        self.w_ = self.rgen_.normal(loc=0.0, scale=0.01, size=m)
        self.b_ = np.float64(0)
        self.initialized_weights = True
    def _update_weights(self, xi, target):
        output = self.activation(self.net_input(xi))
        error = target - output
        self.w_ += self.eta * 2.0 * error * xi
        self.b_ += self.eta * 2.0 * error
        loss = error**2
        return loss
    def _shuffle(self, X, y):
        r = self.rgen_.permutation(len(y))
        return X[r], y[r]
    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_
    def activation(self, X):
        return X
    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)