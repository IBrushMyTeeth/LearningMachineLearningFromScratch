import torch as th
import torch.nn as nn
from simple_XOR_network import Simple_XOR_Network
import matplotlib.pyplot as plt
import numpy as np

# add random seed to make results reproducable
th.manual_seed(19)


# create the XOR tables the model is going to learn
# these are the four possible binary inputs and their XOR labels
X = th.tensor([[0,0], [1,0], [1,1], [0,1]], dtype= th.float32)
y = th.tensor([0, 1, 0, 1], dtype= th.float32).unsqueeze(1)

# we dont need to shuffle or use mini-batches of the dataset neither here nor on iteration
# this is because we already have all possible combinations and their labels
# in other words we dont risk generalization
# XOR is just too small for these techniques


# create the model
model = Simple_XOR_Network()

# set up the optimizer
optimizer = th.optim.SGD(model.parameters(), lr=0.1)

# choose a criterion/loss function
# the XOR problem is in essence a binary classification task
# so binary cross_entropy can be used
criterion = nn.BCELoss()

# train the model
num_iterations = 2500


for i in range(num_iterations):

    # forward pass
    pred = model(X)

    # compute loss
    loss = criterion(pred, y)

    # backward pass and adjust parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
# just use ChatGPT to draw the plots
# drawing them is not the important part, but they 
# visualize what we really are doing


# create a dense grid of points in the 0-1 range
grid_x, grid_y = np.meshgrid(
    np.linspace(0, 1, 200),
    np.linspace(0, 1, 200)
)

# stack the grid points and convert to torch tensor
grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
grid_t = th.tensor(grid_points, dtype=th.float32)

# get model predictions on the grid
with th.no_grad():
    preds = model(grid_t).numpy().reshape(200, 200)

# plot
plt.figure(figsize=(6,6))

# decision regions as heatmap
plt.contourf(grid_x, grid_y, preds, levels=50, cmap="viridis", alpha=0.8)

# decision boundary line (where output = 0.5)
plt.contour(grid_x, grid_y, preds, levels=[0.5], colors="red", linewidths=2)

# plot the original XOR points
plt.scatter(X[:,0], X[:,1], c=y.squeeze(), cmap="coolwarm", edgecolors="k", s=100)

plt.title("XOR Decision Boundary Learned by the Network")
plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.show()


# extract hidden layer weights and biases
w = model.l1.weight.data.numpy()
b = model.l1.bias.data.numpy()

plt.figure(figsize=(6,6))

# plot the original XOR points
plt.scatter(X[:,0], X[:,1], c=y.squeeze(), cmap="coolwarm", edgecolors="k", s=100)

# define grid for visualization
xx = np.linspace(0, 1, 200)
yy = np.linspace(0, 1, 200)
grid_x, grid_y = np.meshgrid(xx, yy)
grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
grid_t = th.tensor(grid_points, dtype=th.float32)

# compute hidden layer activations
with th.no_grad():
    z1 = model.l1(grid_t)
    a1 = model.a1(z1).numpy()  # shape: (num_points, num_hidden)

# plot each hidden neuron’s activation boundary (activation = 0)
for i in range(a1.shape[1]):
    # reshape neuron activations to grid
    act_grid = a1[:, i].reshape(200, 200)
    # contour where activation crosses 0
    plt.contour(grid_x, grid_y, act_grid, levels=[0], colors=f"C{i+2}", linestyles="--", linewidths=2)

plt.title("Hidden Neuron Boundaries in XOR Network")
plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.show()