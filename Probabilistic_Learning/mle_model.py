import torch as th


class MLE_Model():
    def __init__(self, degress):
        self.degrees = degress

    def extract_features(self, X):
        X_powers = [X ** d for d in range(self.degrees + 1)]
        extracted = th.cat(X_powers, dim=1)
        return extracted
    

    def learn(self, X, labels):
        # setting the derivative of the log likelihood function to
        # zero and solving for the mean actually gives
        # the OLS solution which we know from linear regression
        X = self.extract_features(X)
        # self.weights = th.inverse(X.T @ X) @ X.T @ labels might
        # give errors if the matrix is not invertible
        XT_X = X.T @ X
        self.weights = th.linalg.pinv(XT_X) @ X.T @ labels

        # in a probabilistic framework the sample variance is calculated too
        self.var = 1 / X.shape[0] * ((labels - X @ self.weights)**2).sum()

    def forward(self, X):
        # the forward method returns a distribution
        # which is called the predictive distribution
        X = self.extract_features(X)

        # compute mean
        mean = X @ self.weights
        # var is assumed to be global in normal MLE
        var = self.var

        return mean, var
