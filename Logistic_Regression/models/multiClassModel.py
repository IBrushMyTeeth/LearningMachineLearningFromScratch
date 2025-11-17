import torch


class MultiClassModel(torch.nn.Module):
    def __init__(self, n_features, n_classes, lambda_coef = 0, p = 0):
        super().__init__()
        # the weights.shape = (n_features, n_classes)
        # each column of in weights corresponds to one class
        # the model learns one linear regression for each class
        self.weights = torch.nn.Parameter(torch.rand((n_features,n_classes)))
        # the bias can be 1D it will automatically be broadcasted
        self.bias = torch.nn.Parameter(torch.zeros(n_classes))
        self.lambda_coef = lambda_coef
        self.p = p

    def forward(self, X):
        return X @ self.weights + self.bias
    
    def learn(self, X, labels, steps = 10000, tol=1e-6):
        # This time use adam (momentum-based optimizer)instead of sgd
        # adam adds an element, velocity to gradient descent 
        # adam works better for small sets and has a faster convergence
        optimizer = torch.optim.Adam(self.parameters(), lr= 0.01)
        prev_loss = float("inf")

        for i in range(steps):
            # calculate logits
            logits = self.forward(X)

            # set up the loss function + regulization
            loss = torch.nn.functional.cross_entropy(logits,labels)
            if self.p > 0:
                loss += (self.weights.abs()**self.p).sum()*self.lambda_coef

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Early stopping based on tolerance
            if abs(prev_loss - loss.item()) < tol:
                print(f"Stopped early at step {i}, Loss = {loss.item():.6f}")
                break

            prev_loss = loss.item()
    
    def predict(self, X):
        logits = self.forward(X)
        return torch.argmax(logits, dim=1)
