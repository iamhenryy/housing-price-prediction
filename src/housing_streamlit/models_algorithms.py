import numpy as np

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=1))

class KNN_regressor:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

    def predict(self, X):
        X = np.array(X)
        preds = []
        for x in X:
            dists = euclidean_distance(self.X, x)
            idx = np.argsort(dists)[:min(self.k, len(self.X))]
            preds.append(np.mean(self.y[idx]))
        return np.array(preds)

class K_means:
    def __init__(self, k):
        self.k = k

    def fit(self, X):
        self.X = np.array(X, dtype=float)

    def init_centers(self):
        np.random.seed(42)
        idx = np.random.choice(len(self.X), self.k, replace=False)
        self.centers = self.X[idx]

    def assign_labels(self, X):
        X = np.array(X, dtype=float)
        return np.array([
            np.argmin(np.sum((self.centers - x) ** 2, axis=1))
            for x in X
        ])

    def update_centers(self):
        self.labels = self.assign_labels(self.X)
        self.centers = np.array([
            self.X[self.labels == i].mean(axis=0)
            if np.any(self.labels == i)
            else self.X[np.random.randint(len(self.X))]
            for i in range(self.k)
        ])

    def fit_loop(self, max_iter=300):
        self.init_centers()
        for _ in range(max_iter):
            old = self.centers.copy()
            self.update_centers()
            if np.allclose(old, self.centers):
                break
