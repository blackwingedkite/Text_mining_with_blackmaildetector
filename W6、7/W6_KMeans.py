import pandas as pd
dataframe = pd.read_excel('./imdb_70.xlsx')
dataframe.head(5)
labels = dataframe['genre_id']
texts = dataframe['synopsis']

from sklearn.feature_extraction.text import TfidfVectorizer
TFIDF_vectorizer = TfidfVectorizer(min_df=2, stop_words='english')
TFIDF_vectors = TFIDF_vectorizer.fit_transform(texts)
print(TFIDF_vectors.shape)

from sklearn.cluster import KMeans
n_clusters = 16
cost = []
for i in range(2,n_clusters):    
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(TFIDF_vectors)
    cost.append(kmeans.inertia_)

import matplotlib.pyplot as plt
plt.plot(cost, 'bx-')
final_n_clusters = 6
final_kmeans = KMeans(final_n_clusters)
final_kmeans.fit(TFIDF_vectors)
print(final_kmeans.labels_)

from sklearn import metrics
predicted_labels = final_kmeans.labels_
metrics.adjusted_rand_score(labels, predicted_labels)

order_centroids = final_kmeans.cluster_centers_.argsort()[:,::-1]

print("----  Top Terms of Each Cluster ----")
for i in range(final_n_clusters):
    print("\n\nCluster %d keywords: " % i)
    for ind in order_centroids[i, :6]:
        print(TFIDF_vectorizer.get_feature_names()[ind])