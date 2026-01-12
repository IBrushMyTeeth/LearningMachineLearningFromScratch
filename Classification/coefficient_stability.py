from sklearn.datasets import load_breast_cancer
from binaryModel import BinaryModel
import torch as th
import matplotlib.pyplot as plt

# The goal of this module is to study the variance in coefficients
# while trained on different sets


# load the data and initialize variables
data = load_breast_cancer()
X = th.tensor(data.data, dtype= th.float32)
y = th.tensor(data.target, dtype= th.long)

num_rounds = 50
num_features = X.shape[1]
num_samples = X.shape[0]

# initiate the array which will hold each rounds coefficients
collected_coefficients = th.zeros((num_rounds, num_features))


# in each iteration, bootstrap data to simulate infinite data
# train a model
# collect the weights
for i in range(num_rounds):
    # for randint replacement = True by default
    idx = th.randint(0, num_samples, (num_samples, ))
    mini_x = X[idx]
    mini_y = y[idx]

    model = BinaryModel(num_features, lambda_coef= 0.01,p=2)
    model.learn(mini_x, mini_y, tol= 1e-5)

    with th.no_grad():
        weights = model.weights.flatten()
        collected_coefficients[i] = weights


# summary statistics
# can be printed or plotted
coefficient_means = th.mean(collected_coefficients, dim=0)
coefficient_max = th.max(collected_coefficients, dim=0)
coefficient_min = th.min(collected_coefficients, dim=0)
coefficient_variance = th.var(collected_coefficients, dim=0)
coefficient_std = th.std(collected_coefficients, dim=0)


