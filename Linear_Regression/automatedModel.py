import torch

class LinearModel(torch.nn.Module):
    def __init__(self, in_features, out_features, lambda_coef= 0.01, p= 0):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, out_features, bias=True, dtype=torch.float32)
        self.lambda_coef = lambda_coef
        self.p = p

    def forward(self, X):
        return self.fc(X)
    
    def learn(self, X, labels, steps= 10000):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

        for i in range(steps):
            logits = self.forward(X)
            loss = (torch.nn.functional.mse_loss(logits, labels) +
                torch.sum(self.fc.weight.abs()**self.p) * self.lambda_coef)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

