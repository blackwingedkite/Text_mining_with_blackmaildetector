import xlrd
workbook = xlrd.open_workbook('D:/Coding/text_mining/blog-gender-dataset.xls')
booksheet = workbook.sheet_by_name('data')
texts = []
labels = []
for i in range(booksheet.nrows):
    labels.append(booksheet.cell(i,1).value)
    texts.append(booksheet.cell(i,0).value)

# Tokenization
from sklearn.feature_extraction.text import CountVectorizer
TF_vectorizer = CountVectorizer()  # 具體化
TF_vectors = TF_vectorizer.fit_transform(texts)  # 把資料轉換成Sklearn type
# print(TF_vectors)：成為CountVectorizer的形狀：(a,b) c： a是第幾筆檔案，b是在a檔案裏面出現過的字，是字典裏面的第幾個，C是出現過幾次
# 字典：由A到Z排出所有曾經出現過的字詞

x_train = TF_vectors[0:2500]
x_test = TF_vectors[2500:]
y_train = labels[0:2500]
y_test = labels[2500:]
print(y_train.shape)  # 左邊是總共有幾筆資料，右邊是字典的大小(由TF_Vectorizer得知)
print(y_test.shape)

# 計算
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
model.fit(x_train,y_train)  # 利用前2500筆資料來建立模型，那些是F那些是M
predicted_results = []
predicted_results.extend(model.predict(x_test))  # 利用後700筆資料來預測性別

# 呈現
from sklearn import metrics
print(metrics.classification_report(y_test, predicted_results))
print(model.predict_proba(x_test[-1]))  # 看第三筆資料的M跟F分別機率有多高
print(model.predict(x_test[-1]))
print(y_test[-1])
# Precision：被選進這個分類的資料裡面，真的是F的機率有多高
# Recall：真的是F的人裡面，被選進這個資料的機率有多高
# F1-Score：兩者的調和平均數
# M：Precision高，Recall低：被選進這個分類的資料少，但是準確度高
# F：反之。