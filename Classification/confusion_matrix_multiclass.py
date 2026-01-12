from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from multiClassModel import MultiClassModel
import matplotlib.pyplot as plt
import torch as th

# in this module the confusion matrix of a multiclass model will be plotted

# load data
data = load_iris()
X = data.data
y = data.target

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.4, random_state= 33)

# convert to tensors
X_train = th.tensor(X_train, dtype= th.float32)
X_test = th.tensor(X_test, dtype= th.float32)
y_train = th.tensor(y_train, dtype= th.long)
y_test = th.tensor(y_test, dtype= th.long)

# create model
model = MultiClassModel(4, 3, 0.1, 2)
model.learn(X_train, y_train)

# predict on test data
predictions = model.predict(X_test)

# get the confusion matrix
cm = confusion_matrix(y_test, predictions)

# plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.show()