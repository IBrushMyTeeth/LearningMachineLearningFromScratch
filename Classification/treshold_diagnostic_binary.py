from data_utils import get_breast_cancer_dataset
from binaryModel import BinaryModel
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import torch as th

# in this module a threshold-based diagnostic will be given
# breast_cancer is an imbalanced dataset so
# precision will be plotted against recall

X_train, X_test, y_train, y_test = get_breast_cancer_dataset(test_size= 0.5, random_state= 13)

# convert to tensors
X_train = th.tensor(X_train, dtype=th.float32)
X_test = th.tensor(X_test, dtype=th.float32)
y_train = th.tensor(y_train, dtype=th.float32)
y_test = th.tensor(y_test, dtype=th.float32)

# train model
model = BinaryModel(30, 0.01, 2)
model.learn(X_train, y_train)

# get probabilities
probs = model.probabilities(X_test)

# convert back into numpy array
y_test = y_test.detach().numpy()
probs = probs.detach().numpy()

# Compute precision–recall pairs
precision, recall, thresholds = precision_recall_curve(y_test, probs)

# compute AUC
pr_auc = auc(recall[::-1], precision[::-1])


# The are is close to 1, which means that the model performs very well
print(pr_auc)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.grid(True)
plt.show()