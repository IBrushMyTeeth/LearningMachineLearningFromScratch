import torch as th

class BayesianPolyReg:
    def __init__(self, degrees, alpha=1, sigma=0.1):
        self.alpha = alpha
        self.sigma = sigma
        self.degrees = degrees

    def extract_features(self, X):
        X_powers = [X ** d for d in range(self.degrees + 1)]
        extracted = th.cat(X_powers, dim=1)
        return extracted
    

    def learn(self, X, y):

        I = th.eye(X.shape[1])
        # compute the posterior precision matrix following the formula
        A = self.alpha * I + (self.sigma **-2) * (X.T @ X)

        # now posterior covariance matrix and posterior mean can be gotten
        self.posterior_covar = th.linalg.inv(A)
        self.posterior_mean = (self.sigma**-2) * self.posterior_covar @ X.T @ y


    def forward(self, X):
        # like all linear regression models the mean is X @ w
        predicted_mean = X @ self.posterior_mean

        model_var = th.sum((X @ self.posterior_covar) * X, dim=1)
        # total variance = model uncertainty + sigma
        predicted_var = model_var + self.sigma**2

        return predicted_mean, predicted_var