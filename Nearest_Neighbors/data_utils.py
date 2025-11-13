from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# load and split data
def load_moons_dataset(test_size=0.3, noise=0.3, random_state=42):
    X, y = make_moons(n_samples=1000, noise=noise, random_state=random_state)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
