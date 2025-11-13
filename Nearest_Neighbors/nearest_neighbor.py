import numpy as np

class KNearestNeighbor:
    def __init__(self, X_reference, labels, k):
        # save reference data
        self.X_reference = X_reference
        self.labels = labels
        self.k = k
    
    def predict(self, X):
        # count samples
        n_samples = X.shape[0]

        # create an empty list of predictions
        y_pred = np.zeros(n_samples, dtype= np.int64)

        for i in range(n_samples):
            # each row corresponds to a sample
            current_sample = X[i, :]

            # calculate euclidian distance, manhattan distance is an alternative
            # for larger datasets use np.linalg.norm which is faster
            distances = np.sqrt(np.sum((self.X_reference - current_sample)**2, axis= 1))

            # get the nearest neighbors
            neighbors = np.argsort(distances)[:self.k]

            # retrieve labels of the neighbors
            neighbor_labels = self.labels[neighbors]

            # decide class using majority voting
            count = np.bincount(neighbor_labels)
            majority_label = np.argmax(count)

            y_pred[i] = majority_label
        return y_pred
