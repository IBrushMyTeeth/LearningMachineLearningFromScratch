from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import torch
from breast_cancer_classifier import BreastCancerClassifier

# load and split the data
data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 1) 

# Convert data into Torch format
X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
y_train = torch.tensor(y_train).float()
y_test = torch.tensor(y_test).float()

# make sure the labels are column vectors
# IMPORTANT: The shape of labels led to very bad accuracy because
# Pytorch does not raise error for shape mismatch
y_train = y_train.view(-1, 1)
y_test = y_test.view(-1, 1)


# scaling using mean and std from training set
# This prevents data leakage from test set to training set
mean = torch.mean(X_train, axis= 0)
std = torch.std(X_train, axis= 0)

X_train = (X_train - mean) / std

X_test = (X_test - mean) / std

# create a model with lambda = 0.1 and Ridge regulization
model = BreastCancerClassifier(30, 0.1, 2)

# train the model
model.learn(X_train, y_train)

# test the model
predictions = model.is_cancer(X_test)

# convert the diagnosis and test labels to 1D arrays
# This makes comparing them easier

accuracy = (predictions == y_test).float().mean()
print(f"Accuracy = {accuracy: .5f}")

# manual confusion matrix
tp = torch.sum((predictions == 1) & (y_test == 1))
fp = torch.sum((predictions == 1) & (y_test == 0))
tn = torch.sum((predictions == 0) & (y_test == 0))
fn = torch.sum((predictions == 0) & (y_test == 1))