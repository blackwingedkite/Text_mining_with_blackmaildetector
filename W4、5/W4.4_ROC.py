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
TF_vectors = TF_vectorizer.fit_transform(texts)

x_train = TF_vectors[0:2500]
x_test = TF_vectors[2500:]
y_train = labels[0:2500]
y_test = labels[2500:]

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
model.fit(x_train,y_train)
# ff = model.predict_proba(x_test)  # predict.proba：是F跟是M的機率各是多少(加起來是1)
# print(ff)
y_probs = model.predict_proba(x_test)[:,0]  # 每一個檔案是F的機率是多少
from sklearn.metrics import roc_curve
from sklearn.metrics import auc  # area under curve
print(type(y_test))
print(type(y_probs))
fpr, tpr, _ = roc_curve(y_test, y_probs, pos_label="F")
# print(fpr)  # false positive rate,電腦以為是女生其實是男生的機率，從小到大排序
# print(tpr)  # True positive rate, 電腦以為是女生，也真的是女生的機率
auc_score = auc(fpr, tpr)
print(auc_score)


import matplotlib.pyplot as plt
plt.figure(1)
plt.plot([0,1], [0,1], "k--")  # 45度曲線
plt.plot(fpr,tpr, label='test (AUC=%0.2f)' %  auc_score)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc='best')  # 調整位置
plt.savefig('ROC_Curve.svg')
