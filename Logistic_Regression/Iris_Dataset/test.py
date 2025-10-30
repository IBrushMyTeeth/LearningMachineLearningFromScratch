from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
import numpy
from iris_plant_classifier import IrisFlowerClassifier


# load and split the data
data = load_iris()
X = data.data
labels = data.target


# The data is relatively simple and well-structured
# therefore a logsitic regression model is expected to perform well
# An older version with test_size = 0,3 almost didnt make any mistakes
# The training set has therefore been reduced to 0,5
# Only for fun and experimenting around
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size= 0.5, random_state= 17)

# convert into torch format
# IMPORTANT this dataset uses doubles
# So convert dtype to float32 which is the standard in pytorch
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

# Convert labels dtype to long
# pytorch's cross entropy expects longs as labels
y_train = torch.tensor(y_train, dtype= torch.long)
y_test =torch.tensor(y_test, dtype=torch.long)

# make sure the labels ARE 1D
# pytorch's cross_entropy expects 1D arrays
y_train = y_train.view(-1)
y_test = y_test.view(-1)

# z-score normalization
mean = torch.mean(X_train, axis=0)
std = torch.std(X_train, axis=0)

X_train = (X_train - mean)/std
X_test = (X_test - mean)/std

model = IrisFlowerClassifier(lambda_coef= 0.01, p= 2)
# The model converges very fast, no need for 10000 steps 
model.learn(X_train, y_train, steps= 1000)
predictions = model.predict(X_test)

accuracy = torch.mean((predictions == y_test).float())
print(f"Accuracy = {accuracy:.2f}")

# Drawing a roc-curve for versicolor vs rest

# for plotting setosa vs rest, get all rows where true label = setosa
# setosa = 0, versicolor = 1, virginia = 2
# Remember to convert tensors into arrays
# Sklearn uses np arrays
y_binary_versicolor = (y_test == 1).detach().numpy()
y_probs_versicolor = (model.forward(X_test)[:, 1]).detach().numpy()
fpr, tpr, thresholds = roc_curve(y_binary_versicolor, y_probs_versicolor)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')  # random chance line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Versicolor vs Rest ROC Curve')
plt.legend(loc="lower right")
plt.show()


# confusion matrix
# Convert tensors into numpy arrays
y_test = y_test.detach().numpy()
predictions = predictions.detach().numpy()


cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(cm, display_labels=data.target_names)
disp.plot(cmap="Blues")
plt.show()
