import torch as th
import torch.nn as nn
from simple_XOR_network import Simple_XOR_Network
import matplotlib.pyplot as plt

# add random seed to make results reproducable
# try different seeds, what weights the model initializes with are
# very determining
# random seed 27 is for example stuck and needs many iterations to escape
th.manual_seed(27)


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

# array to collect loss per iteration
loss_collection = th.zeros(num_iterations)

for i in range(num_iterations):

    # forward pass
    pred = model(X)

    # compute loss
    loss = criterion(pred, y)

    # backward pass and adjust parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # record loss
    with th.no_grad():
        loss_collection[i] = criterion(pred, y)

        if i % 200 == 0:
            print(f"Predictions at iteration {i}: {pred.squeeze().tolist()}")
    

# plot loss over iterations
plt.figure(figsize=(6,4))
plt.plot(loss_collection.numpy())
plt.title("Training Loss Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()