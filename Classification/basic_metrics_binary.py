import torch as th
from binaryModel import BinaryModel
from data_utils import get_breast_cancer_dataset

# In this module the basic metrics will be manually calculated
# The dataset used is Breast Cancer Wisconsin dataset


# load, split, and standardize data
X_train, X_test, y_train, y_test = get_breast_cancer_dataset(test_size= 0.3, random_state= 33)

# convert to tensors
X_train = th.tensor(X_train, dtype=th.float32)
X_test = th.tensor(X_test, dtype=th.float32)
y_train = th.tensor(y_train, dtype=th.float32)
y_test = th.tensor(y_test, dtype=th.float32)


# train the model
model = BinaryModel(30, lambda_coef=0.01, p=2)
model.learn(X_train, y_train)

# predict using learned weights
predictions = model.predict(X_test)

# manual basic meetric calculation
accuracy = ((y_test == predictions).float()).mean()

tp = ((y_test == 1) & (predictions == 1)).float().sum()
fp = ((y_test == 0) & (predictions == 1)).float().sum()

tn = ((y_test == 0) & (predictions == 0)).float().sum()
fn = ((y_test == 1) & (predictions == 0)).float().sum()

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = (2 * precision * recall) / (precision + recall)

# the scores are going to be almost equal because the model
# performs very well on this simple dataset
print(f"Models accuracy is {accuracy:.3f}")
print(f"Models Precision is {precision:.3f}")
print(f"Models Recall is {recall:.3f}")
print(f"Models F1 score is {f1_score:.3f}")
