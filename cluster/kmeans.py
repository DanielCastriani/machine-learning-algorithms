import numpy as np
from utils import to_ndarray
from utils.math_functions import euclidian_distance


class KMeans:

    k: int
    x: np.ndarray
    labels: np.ndarray
    gen_mode: str
    centroids: np.ndarray

    def __init__(self, k=3, gen_mode='rnd_element'):
        """
        Args:
            k (int, optional): number of centroids. Defaults to 3.
            gen_mode (str, optional): centroid generation mode. Defaults to 'rnd_element'.
                * 'rnd_element' will select k elements randomly.
                * 'rnd_values' generate random values between min and max of elements for each centroid
        """

        self.k = k
        self.gen_mode = gen_mode
        self.labels = np.array([])
        self._verify_params()

    def _verify_params(self):
        if self.gen_mode not in ['rnd_element', 'rnd_values']:
            raise Exception("incorrect centroid generation mode. available modes: 'rnd_element', and 'rnd_values'")

    def _init_centroids_random(self):
        min_values = self.x.min(axis=0)
        max_values = self.x.max(axis=0)

        centroid = []

        for i in range(self.k):
            centroid.append([])
            for j in range(self.x.shape[1]):
                centroid[i].append(np.random.uniform(low=min_values[j], high=max_values[j]))

        return np.array(centroid)

    def _init_centroids_by_select(self):
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

    def _init_centroids(self):
        if self.gen_mode == 'rnd_element':
            return self._init_centroids_by_select()
        elif self.gen_mode == 'rnd_values':
            return self._init_centroids_random()

        raise Exception("incorrect centroid generation mode. available modes: 'rnd_element', and 'rnd_values'")

    def _define_labels(self, elements):
        labels = []

        for element in elements:
            distances = euclidian_distance(element, self.centroids)
            labels.append(np.argmin(distances))

        return np.array(labels)

    def _calculate_centroids(self):
        centroids = []

        for index in range(self.k):
            mask = self.labels == index
            selected = self.x[mask]
            mean = np.mean(selected, axis=0)
            centroids.append(mean)

        return np.array(centroids)

    def fit(self, x, max_iter=1000):
        self._verify_params()

        self.x = to_ndarray(x)
        self.centroids = self._init_centroids()

        while max_iter > 0:
            max_iter -= 1

            labels = self._define_labels(self.x)
            if np.array_equal(labels, self.labels):
                break
            else:
                self.labels = labels
                self.centroids = self._calculate_centroids()

    def predict(self, x):
        values = to_ndarray(x)
        return self._define_labels(values)
