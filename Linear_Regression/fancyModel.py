import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class LinearModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, out_features, bias=True, dtype=torch.float32)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

    def forward(self, X):
        return self.fc(X)
    
    def learn(self, X, labels, steps= 10000):
        for i in range(steps):
            logits = self.forward(X)
            loss = torch.nn.functional.mse_loss(logits, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

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

# For each iteration do a gradient descent step and save the errors
for i in range(num_iterations):
    model.learn(X_train, y_train, steps=1)
    
    with torch.no_grad():
        training_pred = model.forward(X_train)
        test_pred = model.forward(X_test)

        train_errors[i] = torch.sqrt(torch.nn.functional.mse_loss(training_pred, y_train))
        test_errors[i] = torch.sqrt(torch.nn.functional.mse_loss(test_pred, y_test))


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
