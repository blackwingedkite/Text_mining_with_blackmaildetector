from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
source = ["Kites have a long and varied history and many different types are flown individually and at festivals worldwide", "Kites may be flown for recreation, art or other practical uses.", "Sport kites can be flown in aerial ballet, sometimes as part of a competition.", "Power kites are multi-line steerable kites designed to generate large forces which can be used to power activities such as kite surfing, kite landboarding,  kite buggying and snow kiting"]

  # TF_Vectors
TF_vectorizer = CountVectorizer()
TF_vectors = TF_vectorizer.fit_transform(source)
print(TF_vectors)  # 第一次轉變為他的format
print(TF_vectors.shape)
print(TF_vectorizer.get_feature_names_out())  # 把東西轉成我們看得懂的原始資料
print(TF_vectors.toarray())  # multiple hot vector，但是有累積次數

  # multiple-hot vectors
binary_vectorizer = CountVectorizer(binary = True)
binary_vectors = binary_vectorizer.fit_transform(source)
# print(binary_vectors.toarray())

  # TF-IDF Vectors
from sklearn.feature_extraction.text import TfidfVectorizer
TFIDF_vectorizer = TfidfVectorizer()  # 實體化
TFIDF_vector = TFIDF_vectorizer.fit_transform(source)
print(TFIDF_vector.toarray())
print(TFIDF_vector)
print(TFIDF_vector.size)
print(TFIDF_vector.shape)
bag_of_words = Counter(source)

  # Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity  # 會回傳一個圖表。
#print(cosine_similarity(TF_vectors,TF_vectors))  # 四個文本之間的關聯性。
#print(cosine_similarity(TF_vectors[0], TF_vectors[2]))
print(cosine_similarity(TF_vectors[0], TF_vectors[2]).flatten()[0])  # 將Table裡面的值取出

  # Euclidean_distances
from sklearn.metrics.pairwise import euclidean_distances
#print(euclidean_distances(TF_vectors,TF_vectors))