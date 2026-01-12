import torch as th

class PolynomialModel(th.nn.Module):
    def __init__(self, degree, input_dim=1):
        super().__init__()
        self.degree = degree
        self.input_dim = input_dim

        # number of features: bias + powers
        num_features = 1 + degree * input_dim

        self.weights = th.nn.Parameter(
            th.randn(num_features, 1) * 0.1
        )

    def extract_features(self, X):
        features = [th.ones((X.shape[0], 1), device=X.device, dtype=X.dtype)]
        for m in range(1, self.degree + 1):
            features.append(X ** m)
        return th.hstack(features)

    def forward(self, X):
        return self.extract_features(X) @ self.weights