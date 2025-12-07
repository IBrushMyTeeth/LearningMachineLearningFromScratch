import torch as th
import torch.nn as nn

# define the neural network

class Simple_XOR_Network(nn.Module):
    def __init__(self):
        super().__init__()
        # first linear layer takes 2 inputs and produces 2 hidden neuron outputs
        # a single linear layer cannot solve XOR, so we need a hidden layer
        self.l1 = nn.Linear(2, 2, True)

        # tanh activation introduces non-linearity
        # this "bends" the input space, allowing the network to separate the XOR pattern
        # which no straight line can do
        self.a1 = nn.Tanh()

        # the second linear layer combines the hidden neurons into a single output
        # this output is still unbounded at this point and can be any real number
        self.l2 = nn.Linear(2, 1, True)

        # sigmoid activation converts the output into the range 0-1
        # this is standard for binary classification
        self.a2 = nn.Sigmoid()
    
    def forward(self, X):
        # pass the input through the layers
        # and apply the activation functions
        z1 = self.a1(self.l1(X))
        z2 = self.a2(self.l2(z1))
        return z2
