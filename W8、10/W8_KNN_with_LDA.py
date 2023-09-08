import xlrd
workbook = xlrd.open_workbook('./blog-gender-dataset.xlsx')
booksheet = workbook.sheet_by_name('data')
texts = []
labels = []

for i in range(booksheet.nrows):
    labels.append(booksheet.cell(i,1).value)
    texts.append(booksheet.cell(i,0).value)

from sklearn.feature_extraction.text import CountVectorizer
TF_vectorizer = CountVectorizer(min_df=1, stop_words='english')
TF_vectors = TF_vectorizer.fit_transform(texts)
print(TF_vectors.shape)

from sklearn.decomposition import LatentDirichletAllocation as LDA
lda_model = LDA(n_components = 10)
LDA_vectors = lda_model.fit_transform(TF_vectors)
print(LDA_vectors)
print(LDA_vectors.shape)
print(lda_model.components_.shape)

def print_topics(model, vectorizer, n_top_words):
    words = vectorizer.get_feature_names()
    for topic_index, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_index)
        print(" ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
        #print(topic.argsort()[:-n_top_words - 1:-1])

print_topics(lda_model, TF_vectorizer, 10)

x_train = LDA_vectors[0:2500]
x_test = LDA_vectors[2500:]
y_train = labels[0:2500]
y_test = labels[2500:]

from sklearn.neighbors import KNeighborsClassifier
KNN_model = KNeighborsClassifier(n_neighbors = 5)
KNN_model.fit(x_train,y_train)

predicted_results = []
expected_results = []
expected_results.extend(y_test)
predicted_results.extend(KNN_model.predict(x_test))
print(predicted_results)

from sklearn import metrics
print(metrics.classification_report(expected_results, predicted_results))