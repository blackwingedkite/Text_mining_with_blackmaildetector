import xlrd
workbook = xlrd.open_workbook('D:/Coding/text_mining/blog-gender-dataset.xls')
booksheet = workbook.sheet_by_name('data')
texts = []
labels = []
for i in range(booksheet.nrows):
    labels.append(booksheet.cell(i,1).value)
    texts.append(booksheet.cell(i,0).value)

from sklearn.feature_extraction.text import CountVectorizer
binary_vectorizer = CountVectorizer(binary=True)  # 原本會計算次數，現在不會了
binary_vectors = binary_vectorizer.fit_transform(texts)
print(binary_vectors.shape)
x_train = binary_vectors[0:2500]
x_test = binary_vectors[2500:]
y_train = labels[0:2500]
y_test = labels[2500:]
print(type(x_train))
print(type(y_train))
from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB()  # 他的方式很不一樣，是用binary，所有F的句子裡面有幾個出現單詞A
model.fit(x_train,y_train)
predicted_results = []
predicted_results.extend(model.predict(x_test))

from sklearn import metrics
print(metrics.classification_report(y_test, predicted_results))
print(model.predict_proba(x_test[-1]))  # 看第三筆資料的M跟F分別機率有多高
print(model.predict(x_test[-1]))
print(type(x_test[-1]))