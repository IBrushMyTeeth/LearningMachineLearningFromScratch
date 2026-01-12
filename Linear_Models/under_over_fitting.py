import torch as th
from polynomialModel import PolynomialModel
import matplotlib.pyplot as plt

# random seed for reproducability
th.manual_seed(11)

# generate synthetic data
num_samples = 50
inputs = th.linspace(0, 1, num_samples).unsqueeze(1)
outputs = th.sin(2 * th.pi * inputs)

# add noise
labels = outputs + 0.25 * (th.rand(num_samples, 1) - 0.5)

# split into train/test
num_samples_train = 25
idx = th.randperm(num_samples)

X_train = inputs[idx[:num_samples_train]]
X_test = inputs[idx[num_samples_train:]]
y_train = labels[idx[:num_samples_train]]
y_test = labels[idx[num_samples_train:]]

# data collection and initialization
max_degrees = 20
max_iter = 3000
train_loss = th.zeros((max_degrees))
test_loss = th.zeros((max_degrees))

criterion = th.nn.MSELoss(reduction="mean")
# train and record loss
for m in range(max_degrees):
    # reseed for fairness
    th.manual_seed(11)

    model = PolynomialModel(m)
    optimizer = th.optim.Adam(model.parameters(), lr=0.01)

    for i in range(max_iter):
        # forward pass
        y_pred = model(X_train)

        # compute loss
        loss = criterion(y_pred, y_train)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # record loss
    with th.no_grad():
        train_loss[m] = th.sqrt(criterion(model(X_train), y_train))
        test_loss[m] = th.sqrt(criterion(model(X_test), y_test))

degrees = range(max_degrees)

plt.figure(figsize=(7, 5))
plt.plot(degrees, train_loss.numpy(), marker="o", label="Train RMSE")
plt.plot(degrees, test_loss.numpy(), marker="o", label="Test RMSE")

plt.xlabel("Polynomial Degree")
plt.ylabel("RMSE")
plt.title("Training vs Test Loss")
plt.legend()
plt.grid(True)
plt.show()

print(test_loss)
print(train_loss)