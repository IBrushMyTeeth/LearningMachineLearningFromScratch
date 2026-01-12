from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

# The diabetes dataset contains samples from 442 patients
# The features are age, bmi, sec etc.
# The features ARE all scaled
# The label is a continuous value called disease progression

data = load_diabetes()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 7)


class LinearModel:
    # Normal equation will be tried this time instead of gradient descent
    def __init__(self):
        pass

    def extract_features(self, X):
        # The features are already scaled so just add the bias
        ones = np.ones((X.shape[0],1))
        return np.hstack([ones, X])

    def learn(self, X_train, labels):
        # Make sure that labels is a vector
        labels = labels.reshape(-1, 1)
        # Extract features
        X_train = self.extract_features(X_train)
        # Calculate the weights using normal equation
        XtX_inverse = np.linalg.inv((X_train.T @ X_train))
        self.weights = XtX_inverse @ X_train.T @ labels
    
    def predict(self, X_test):
        X_test = self.extract_features(X_test)
        return (X_test @ self.weights).ravel()

model = LinearModel()
model.learn(X_train,y_train)
predictions = model.predict(X_test)

# Compare actual vs predicted
print("R²:", r2_score(y_test, predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test, predictions)))
print("MAE:", mean_absolute_error(y_test, predictions))


# This plots the test
# x_axis = true value of a sample
# y_axis = predicted value of a sample
# if x == y then the prediction was exactly right
plt.scatter(y_test, predictions, alpha=0.7)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted (Linear Model)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()
