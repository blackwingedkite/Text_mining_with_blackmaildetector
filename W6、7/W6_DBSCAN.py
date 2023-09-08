import pandas as pd
dataframe = pd.read_excel('./imdb_70.xlsx')
dataframe.head(5)
labels = dataframe['genre_id']
texts = dataframe['synopsis']
from sklearn.feature_extraction.text import TfidfVectorizer
TFIDF_vectorizer = TfidfVectorizer(min_df=2, stop_words='english')
TFIDF_vectors = TFIDF_vectorizer.fit_transform(texts)
print(TFIDF_vectors.shape)

from sklearn.cluster import DBSCAN 
dbscan = DBSCAN(metric='euclidean', eps=1.2, min_samples=2) 
dbscan.fit(TFIDF_vectors)
print(dbscan.labels_)

from collections import Counter
print(Counter(dbscan.labels_))

from sklearn import metrics
predicted_labels = dbscan.labels_
metrics.adjusted_rand_score(labels, predicted_labels)

from sklearn.neighbors import NearestNeighbors
nearest_neighbors = NearestNeighbors(n_neighbors = 5, metric='euclidean')
nearest_neighbors.fit(TFIDF_vectors)
distances, indices = nearest_neighbors.kneighbors(TFIDF_vectors)

import numpy as np
distances = np.sort(distances, axis=0)[:,1]

import matplotlib.pyplot as plt
plt.plot(distances)