import numpy as np
from utils import to_ndarray


class KMeans:
    k: int
    x: np.ndarray
    centroids: np.ndarray

    def __init__(self, k=3):
        self.k = k

    def _init_centroids(self):
        attempts = 50
        centroids = None

        # Select k random points while not duplicates, or attempts == 0
        while attempts > 0:
            centroids_idx = np.random.choice(len(self.x), size=self.k, replace=False)
            centroids = self.x[centroids_idx]

            if len(np.unique(centroids, axis=0)) == self.k:
                return centroids
            
            attempts -= 1

        print('warn: two or more centroids are equal')

        return centroids

    def fit(self, x):
        self.x = to_ndarray(x)
        self.centroids = self._init_centroids()

        # TODO rm line
        return self.centroids
