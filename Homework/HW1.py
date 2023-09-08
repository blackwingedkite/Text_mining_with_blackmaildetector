from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
import numpy as np
from scipy import sparse
TFIDF_vectorizer = TfidfVectorizer(lowercase = True, stop_words= "english")

big_list = []
for i in range(1095):
    with open(file = 'D:/程式設計/text_mining/PA1-data/'+str(i+1)+'.txt',mode = 'r') as f:
        big_list.append(f.read())

TFIDF_vector = TFIDF_vectorizer.fit_transform(big_list)

for i in range(1095):
    np.save('C:/Users/user/Desktop/B10703049/output/'+str(i+1)+".npy", TFIDF_vector[i].toarray())
np_file1 = np.load("C:/Users/user/Desktop/B10703049/output/1.npy",allow_pickle=True)
np_file2 = np.load("C:/Users/user/Desktop/B10703049/output/2.npy",allow_pickle=True)

sparse_Matrix1=sparse.csr_matrix(np_file1)
sparse_Matrix2=sparse.csr_matrix(np_file2)
print(cosine_similarity(sparse_Matrix1, sparse_Matrix2).flatten()[0])