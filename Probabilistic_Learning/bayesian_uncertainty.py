import torch as th
from bayesian_model import BayesianPolyReg
import matplotlib.pyplot as plt

# generate data
X = th.linspace(0, 2, 40).unsqueeze(1)
y = th.sin(X) + 0.25 * th.rand_like(X)

# train model on all data
model = BayesianPolyReg(2)
X_extracted = model.extract_features(X)
model.learn(X_extracted, y)

# dense grid for smooth plotting
X_plot = th.linspace(0, 2, 200).unsqueeze(1)
X_plot_extracted = model.extract_features(X_plot)

# get predicted mean, var and std
mean, var = model.forward(X_plot_extracted)
std = th.sqrt(var)

# print std -> model has different levels of uncertainty based on X
print(std)

# convert to numpy
X_np = X.numpy().ravel()
y_np = y.numpy().ravel()

X_plot_np = X_plot.numpy().ravel()
mean_np = mean.detach().numpy().ravel()
std_np = std.detach().numpy().ravel()

# plot
plt.figure(figsize=(10,6))

# noisy data
plt.scatter(X_np, y_np, color='blue', alpha=0.6, label='data')

# predictive mean
plt.plot(X_plot_np, mean_np, color='red', label='predictive mean')

# uncertainty band ±1σ
plt.fill_between(
    X_plot_np,
    mean_np - std_np,
    mean_np + std_np,
    color='red',
    alpha=0.2,
    label='±1σ'
)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Bayesian Polynomial Regression Predictions with Uncertainty')
plt.legend()
plt.show()
