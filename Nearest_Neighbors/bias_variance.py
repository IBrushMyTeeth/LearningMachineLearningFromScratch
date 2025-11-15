from data_utils import load_moons_dataset
from nearest_neighbor import KNearestNeighbor
import numpy as np
import matplotlib.pyplot as plt

# Random seed to allow reproducing data
np.random.seed(17)

def compute_bias_variance(model_class, k, X_train, X_test, y_train, y_test, train_size, rounds):

    # each row will save the predictions on test set
    predictions = np.zeros((rounds, len(y_test)))

    for i in range(rounds):
        # randomly choose some indices
        # we only have 1 training set
        # So randomly bootstrap samples from the training data
        # to simulate different training datasets.
        idx = np.random.choice(len(X_train), size=train_size, replace=True)
        sample_X = X_train[idx]
        sample_labels = y_train[idx]

        # train the model on the randomly chosen samples
        model = model_class(sample_X, sample_labels, k)

        pred = model.predict(X_test)

        predictions[i] = pred
    
    # compute the average prediction
    mean_pred = np.mean(predictions, axis= 0)

    # calculate bias, var and total error
    bias_sq = np.mean((mean_pred - y_test)**2)
    var = np.mean(np.var(predictions, axis=0))
    total_error = bias_sq + var

    return bias_sq, var, total_error

# load and split data
X_train, X_test, y_train, y_test = load_moons_dataset()

# initialize
max_k = 50
bias_sq = np.zeros(max_k)
var = np.zeros(max_k)
total_error = np.zeros(max_k)

# for each k, compute variance, bias, total error and save the result
for i in range(1, max_k + 1):
    curr_bias_sq, curr_var, curr_total_error = compute_bias_variance(
        KNearestNeighbor, i, X_train, X_test, y_train, y_test, train_size= 300, rounds= 50)
    
    bias_sq[i - 1] = curr_bias_sq
    var[i - 1] = curr_var
    total_error[i - 1] = curr_total_error


# plot
k_values = np.arange(1, max_k + 1)
plt.figure(figsize=(10, 6))
plt.plot(k_values, bias_sq, label='Bias²', color='red')
plt.plot(k_values, var, label='Variance', color='blue')
plt.plot(k_values, total_error, label='Total Error (Excludes irreducable noise)', color='green')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Error')
plt.title('Bias-Variance Tradeoff for k-NN')
plt.legend()
plt.show()