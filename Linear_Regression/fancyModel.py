import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class LinearModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, out_features, bias=True, dtype=torch.float32)

    def forward(self, X):
        return self.fc(X)
    

# fetch and split the data into train/test
data = fetch_california_housing()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.4, random_state= 23)

# transform into tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1,1)

# scale using mean and std
mean = torch.mean(X_train, dim=0)
std = torch.std(X_train, dim=0)

X_train = (X_train - mean)/std
X_test = (X_test - mean)/std

# create the model
model = LinearModel(8, 1)
num_iterations = 250

# collect errors / these will be plotted
train_errors = torch.zeros(num_iterations)
test_errors = torch.zeros(num_iterations)

# train the model
criterion = torch.nn.MSELoss(reduction="mean")
optimizer = torch.optim.SGD(model.parameters(), lr= 0.01)

for i in range(num_iterations):
    # forward pass
    y_pred = model(X_train)

    # compute loss
    loss = criterion(y_pred, y_train)

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # record loss
    with torch.no_grad():
        train_errors[i] = torch.sqrt(criterion(model(X_train), y_train))
        test_errors[i] = torch.sqrt(criterion(model(X_test), y_test))



# plot the errors
plt.figure(figsize=(8, 5))
plt.plot(train_errors, label="Train Error")
plt.plot(test_errors, label="Test Error")
plt.xlabel("Iteration")
plt.ylabel("RMSE Loss")
plt.title("Training vs Test Error over Iterations")
plt.legend()
plt.grid(True)
plt.show()
