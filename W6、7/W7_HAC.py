import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram


def plot_dendrogram(model, **kwargs):
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py
    # Children of hierarchical clustering
    children = model.children_
    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

dataframe = pd.read_excel('D:/Coding/text_mining/W6、7/imdb_70.xlsx')
dataframe.head(5)
labels = dataframe['genre_id']
texts = dataframe['synopsis']
print(texts[0])

from sklearn.feature_extraction.text import TfidfVectorizer
TFIDF_vectorizer = TfidfVectorizer(min_df=2, stop_words='english')
TFIDF_vectors = TFIDF_vectorizer.fit_transform(texts)
print(TFIDF_vectors.shape)

from sklearn.cluster import AgglomerativeClustering
hac = AgglomerativeClustering(linkage='ward', n_clusters=2)
hac.fit(TFIDF_vectors.toarray())
print(hac.labels_)

from collections import Counter
print(Counter(hac.labels_))

from sklearn import metrics
predicted_labels = hac.labels_
metrics.adjusted_rand_score(labels, predicted_labels)
plt.title('Hierarchical Clustering Dendrogram')
plot_dendrogram(hac, labels=hac.labels_)
plt.show()