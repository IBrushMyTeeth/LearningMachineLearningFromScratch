import torch as th
import torch.nn as nn

# Binary classification model
# This model uses gradient descent from pytorch instead of the previously manual gradient descent
# lambda_coef and p are used for regularization
# p = 0 -> no regularization, p = 1 -> LASSO, p = 2 -> Ridge
class BinaryModel(nn.Module):
    def __init__(self, n_features, lambda_coef, p):
        super().__init__()
        self.weights = nn.Parameter(th.randn((n_features, 1)))
        self.bias = nn.Parameter(th.randn(1))
        self.lambda_coef = lambda_coef
        self.p = p
    
    def forward(self, input):
        return input @ self.weights + self.bias
    
    
    def learn(self, inputs, labels, tol = 1e-6, n_steps= 10000):
        optimizer = th.optim.SGD(self.parameters(), lr= 0.001)
        # make sure labels has the right shape
        labels = labels.view(-1, 1)

        # initiate previous loss as infinity
        prev_loss = float("inf")

        for i in range(n_steps):
            logits = self.forward(inputs)
            probs = th.sigmoid(logits)
            
            # This is a translation of the mathematical loss function for a 01 model, th.
            # nn.functional.cross_entroppy could also be used, this is a manual version for learning purposes
            loss = - (1/inputs.shape[0]) * th.sum(labels * th.log(probs + 1e-8) + \
                                            (1 - labels) * th.log(1-probs + 1e-8)) + \
                                            (self.weights.abs()**self.p).sum()*self.lambda_coef
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # check if improvement is too little
            if abs(prev_loss - loss.item()) < tol:
                break

            prev_loss = loss.item()

    
    def predict(self, inputs):
        # threshold is by default 50% 
        probability = th.sigmoid(self.forward(inputs))
        return (probability >= 0.5).int().flatten()
    
    def probabilities(self, X):
        logits = self.forward(X)
        return th.sigmoid(logits)