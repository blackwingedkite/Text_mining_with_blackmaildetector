import xlrd
workbook = xlrd.open_workbook('text_mining/blog-gender-dataset.xls')
booksheet = workbook.sheet_by_name('data')
texts = []
labels = []
for i in range(booksheet.nrows):
    labels.append(booksheet.cell(i,1).value)
    texts.append(booksheet.cell(i,0).value)
from sklearn.feature_extraction.text import TfidfVectorizer
TFIDF_vectorizer = TfidfVectorizer(min_df=1)
TFIDF_vectors = TFIDF_vectorizer.fit_transform(texts)
TFIDF_vectors.shape

x_train = TFIDF_vectors[0:2500]
x_test = TFIDF_vectors[2500:]
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