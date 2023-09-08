import xlrd
workbook = xlrd.open_workbook('D:/Coding/text_mining/blog-gender-dataset.xls')
booksheet = workbook.sheet_by_name('data')
texts = []
labels = []
for i in range(booksheet.nrows):
    labels.append(booksheet.cell(i,1).value)
    texts.append(booksheet.cell(i,0).value)

from sklearn.feature_extraction.text import CountVectorizer
TF_vectorizer = CountVectorizer()  # 具體化
TF_vectors = TF_vectorizer.fit_transform(texts)  # 把資料轉換成Sklearn type
x_train = TF_vectors[0:2500]
x_test = TF_vectors[2500:]
y_train = labels[0:2500]
y_test = labels[2500:]

from sklearn.model_selection import cross_val_score
from sklearn. naive_bayes import MultinomialNB
model = MultinomialNB()
scores = cross_val_score(model, x_train, y_train, cv=10, scoring='f1_macro')
print(scores.mean())
print(scores)