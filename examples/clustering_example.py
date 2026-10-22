import numpy as np

from minislearn.cluster.kmeans import KMeans


X = np.array([
    [1, 1],
    [2, 2],
    [1, 2],
    [8, 8],
    [9, 9],
    [8, 9]
])


model = KMeans(
    n_clusters=2
)

model.fit(X)

labels = model.predict(X)


print("Cluster labels:", labels)
print("Cluster centers:")
print(model.cluster_centers_)