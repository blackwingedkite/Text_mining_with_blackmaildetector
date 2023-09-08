x_rawdata = []
y_rawdata = []
with open(file = 'D:/Coding/text_mining/training_new.txt',mode = 'r') as f:
    for i in range(1,14):
        nums = f.readline().split(" ")
        for j in nums[1:-2]:
            with open(file = "D:/Coding/text_mining/PA1-data/"+j+".txt") as f2:
                x_rawdata.append(f2.read())
                y_rawdata.append(i)

from sklearn.feature_extraction.text import CountVectorizer
binary_vectorizer = CountVectorizer(binary=True)
binary_vectors = binary_vectorizer.fit_transform(x_rawdata)

from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
model = BernoulliNB()
X_train, X_test, y_train, y_test = train_test_split(binary_vectors, y_rawdata, test_size=0.1, random_state=3, stratify = y_rawdata)
model.fit(X_train,y_train)
predicted_results = []
predicted_results.extend(model.predict(X_test))

from sklearn import metrics
print(metrics.classification_report(y_test, predicted_results))
print(model.predict_proba(X_test[-1]))  # 看第三筆資料的M跟F分別機率有多高

#-------

big_list = []
import csv
file = open('D:/Coding/text_mining/hw2_sam.csv')
reader = csv.DictReader(file)
for row in reader:
    with open(file = "D:/Coding/text_mining/PA1-data/"+row['Id']+".txt") as f3:
        big_list.append(f3.read())
binary_vectors2 = binary_vectorizer.fit_transform(big_list)
predict_result2 = []
predict_result2.extend(model.predict(binary_vectors2))

print(predict_result2)


