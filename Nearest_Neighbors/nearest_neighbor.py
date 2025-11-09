import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class OneNearestNeighbor:
    def __init__(self, X_reference, labels):
        # save reference data
        self.X_reference = X_reference
        self.labels = labels
    
    def predict(self, X):
        # count samples
        n_samples = X.shape[0]

        # create an empty list of predictions
        y_pred = np.zeros(n_samples)

        for i in range(n_samples):
            # each row corresponds to a sample
            current_sample = X[i, :]

            # calculate euclidian distance, manhattan distance is an alternative
            # for larger datasets use np.linalg.norm which is faster
            distances = np.sqrt(np.sum((self.X_reference - current_sample)**2, axis= 1))

            # retrieve label of the nearest neighbor
            y_pred[i] = self.labels[np.argmin(distances)]
        
        return y_pred

# load and spit data
data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 29)

# create a 1NN model
model = OneNearestNeighbor(X_train, y_train)

# predict on X_test
predictions = model.predict(X_test)

# set up a confusion matrix to evaluate performance
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(cm, display_labels=data.target_names)
disp.plot(cmap="Blues")
plt.title("1NN Confusion Matrix")
plt.show()