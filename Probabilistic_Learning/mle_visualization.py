import matplotlib.pyplot as plt
from mle_model import MLE_Model
import torch as th

# Generate data
X = th.linspace(start=0, end=2, steps=200).unsqueeze(1)
y = th.sin(X)
y += th.rand_like(y) * 0.25


# create and train model
# also try it 1 and 3 degrees, its interesting to see
model = MLE_Model(2)
X_extracted = model.extract_features(X)
model.learn(X_extracted, y)

# get the learned mean and var
mean, var = model.forward(X_extracted)
std = th.sqrt(var)

# print std -> model assumes global variance, so models uncertainty doesnt depend on X
print(std)

# for plotting convert to numpy arrays
X = X.numpy().ravel()
y = y.numpy().ravel()
mean = mean.detach().numpy().ravel()
var = var.detach().numpy().ravel()
std = std.detach().numpy().ravel()

# plotting
plt.figure(figsize=(10,6))

# noisy points
plt.scatter(X, y, s=10, color='blue', alpha=0.4, label='noisy data')

# mean prediction
plt.plot(X, mean, color='red', label='mean')

# ±1 std band (var is scalar)
plt.fill_between(
    X, 
    mean - std, 
    mean + std, 
    color='red', alpha=0.2, label='±1σ'
)

plt.legend()
plt.show()