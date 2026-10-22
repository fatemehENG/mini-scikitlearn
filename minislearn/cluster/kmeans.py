import numpy as np

from minislearn.base.estimator import BaseEstimator


class KMeans(BaseEstimator):

    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):

        X = np.asarray(X, dtype=float)

        self.n_features_in_ = X.shape[1]

        indices = np.random.choice(
            X.shape[0],
            self.n_clusters,
            replace=False
        )

        self.cluster_centers_ = X[indices].copy()

        for _ in range(self.max_iter):

            distances = np.linalg.norm(
                X[:, np.newaxis] - self.cluster_centers_,
                axis=2
            )

            labels = np.argmin(
                distances,
                axis=1
            )

            new_centroids = np.array([
                X[labels == i].mean(axis=0)
                for i in range(self.n_clusters)
            ])

            if np.allclose(
                self.cluster_centers_,
                new_centroids
            ):
                break

            self.cluster_centers_ = new_centroids

        self.labels_ = labels

        return self



    def predict(self, X):
        X= np.asarray(X, dtype= float)
        distances= np.linalg.norm(X[:, np.newaxis]- self.cluster_centers_ , axis=2)
        labels= np.argmin(distances, axis=1)
        return labels 

    