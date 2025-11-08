from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt


# The california housing dataset contains features like HouseAge, AveRooms, AveBedrooms, etc.
# The label is a single continues value 
# The features are NOT scaled


# load the dataset
data = fetch_california_housing()
X = data.data
y = data.target

# train and test using 300 samples
x_small = X[:300]
y_small = y[:300]

X_train, X_test, y_train, y_test = train_test_split(x_small, y_small, test_size= 0.3, random_state= 42)


# Create the linear model
class Linear_model:
    def __init__(self):
        pass

    def initialize_weights(self, n_features):
        # this is a linear model so add 1 row for bias
        self.weights = np.zeros((n_features + 1, 1))
    
    def extract_features(self, X):
        # Scale the weights
        self.feature_min = X.min(axis=0)
        self.feature_max = X.max(axis=0)
        X_scaled = (X - self.feature_min) / (self.feature_max - self.feature_min)

        # Add Bias
        ones = np.ones((X_scaled.shape[0], 1))
        return np.hstack([ones, X_scaled])


    def learn(self, X, labels, learning_rate= 0.01, tol=1e-6, max_iterations=10000):
        self.loss_history = []
        X = self.extract_features(X)
        labels = labels.reshape(-1, 1)
        # Apply gradient descent to adjust weights
        for i in range(max_iterations):
            current_error = 0.5 * np.mean((X @ self.weights - labels)**2)

            new_weights = self.weights - (learning_rate * (X.T @ ((X @ self.weights - labels)) / X.shape[0]))
            new_error = 0.5 * np.mean((X @ new_weights - labels)**2)

            self.weights = new_weights

            # if the improvement is too little stop
            if abs(current_error - new_error) < tol:
                break

            if i % 500 == 0:
                self.loss_history.append(current_error)
        print(f"Iterations: {i}")


    def predict(self, X):
        # scale using saved min/max
        scaled = (X - self.feature_min) / (self.feature_max - self.feature_min)

        ones = np.ones((scaled.shape[0], 1))
        extracted_features = np.hstack([ones, scaled])
        return extracted_features @ self.weights


# create a model, train, and test:
model = Linear_model()
model.initialize_weights(X_train.shape[1])
model.learn(X_train, y_train)
prediction = model.predict(X_test)


# Compare actual vs predicted
print("R²:", r2_score(y_test, prediction))
print("RMSE:", np.sqrt(mean_squared_error(y_test, prediction)))
print("MAE:", mean_absolute_error(y_test, prediction))

# Plot training loss over iterations
plt.figure(figsize=(8, 5))
plt.plot(model.loss_history, marker='o', linestyle='-')
plt.title("Training Loss Over Iterations")
plt.xlabel("Iteration (x500 steps)")  # because you log every 500 iterations
plt.ylabel("Mean Squared Error (MSE)")
plt.grid(True)
plt.show()

