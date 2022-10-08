import numpy as np
from sklearn.neighbors import NearestNeighbors
import distances as mtrk


class KNNClassifier:
    def __init__(self, k, strategy, metric, weights, test_block_size):
        self.k = k
        self.strategy = strategy
        self.metric = metric
        self.weights = weights
        self.test_block_size = test_block_size
        if self.strategy == 'brute':
            self.model = NearestNeighbors(n_neighbors=self.k,
                                          algorithm=self.strategy,
                                          metric=self.metric)
        elif self.strategy == 'kd_tree' or self.strategy == 'ball_tree':
            self.model = NearestNeighbors(n_neighbors=self.k,
                                          algorithm=self.strategy,
                                          metric='euclidean')

    def fit(self, X, y):
        if (self.strategy == 'brute' or self.strategy == 'kd_tree' or
           self.strategy == 'ball_tree'):
                self.y_train = y
                self.model.fit(X, y)
        elif self.strategy == 'my_own':
            self.X_train, self.y_train = X, y

    def find_kneighbors(self, X, return_distance):
        if (self.strategy == 'brute' or self.strategy == 'kd_tree' or
           self.strategy == 'ball_tree'):
            return self.model.kneighbors(X, self.k,
                                         return_distance=return_distance)
        elif self.strategy == 'my_own':
            if self.metric == 'euclidean':
                A = mtrk.euclidean_distance(X, self.X_train)
            elif self.metric == 'cosine':
                A = mtrk.cosine_distance(X, self.X_train)
            if return_distance:
                return (np.sort(A, axis=1)[..., :self.k],
                        np.argsort(A, axis=1)[..., :self.k])
            else:
                return np.argsort(A, axis=1)[..., :self.k]

    def predict(self, X):
        if self.weights:
            dists, neighb = self.find_kneighbors(X, return_distance=True)
            A = np.zeros((X.shape[0], np.unique(self.y_train).shape[0]))
            weights = 1/(dists+10**(-5))
            cl = self.y_train[neighb]
            for k in range(self.k):
                for i in range(cl.shape[0]):
                    A[i, cl[i, k]] += weights[i, k]
            return np.argmax(A, axis=1)
        else:
            neighb = self.find_kneighbors(X, return_distance=False)
            answer = []
            for i in self.y_train[neighb]:
                values, counts = np.unique(i, return_counts=True)
                ind = np.argmax(counts)
                answer.append(values[ind])
            return np.array(answer)
