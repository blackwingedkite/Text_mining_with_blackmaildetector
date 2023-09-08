# 讀取1095個檔案
big_list = []
for i in range(1095):
    with open(file = 'D:/coding/text_mining/PA1-data/'+str(i+1)+'.txt',mode = 'r') as f:
        big_list.append(f.read())

# 先將這些檔案Fit_transform至binary vectors
from sklearn.feature_extraction.text import CountVectorizer
binary_vectorizer = CountVectorizer(binary=True)
binary_vectors = binary_vectorizer.fit_transform(big_list)

# 把要被當作training data的195筆資料找出來
from scipy.sparse import vstack
with open(file = 'D:/Coding/text_mining/training_new.txt',mode = 'r') as f:
    # 要先有一個，之後才能繼續加
    x_training = binary_vectors[10]
    y_training = [1]
    nums_row1 = f.readline().split(" ")
    # 把第一行剩下的補完
    for k in nums_row1[2:-1]:
        x_training = vstack((x_training,binary_vectors[int(k)-1]))
        y_training.append(1)
    # 把剩下13行補完
    for i in range(2,14):
        nums = f.readline().split(" ")
        for j in nums[1:-1]:
            x_training = vstack((x_training,binary_vectors[int(j)-1]))
            y_training.append(i)

# 91分配
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_training, y_training, test_size=0.1, random_state=3, stratify = y_training)

# 建構model
from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB()
model.fit(x_train,y_train)
predicted_results = []
predicted_results.extend(model.predict(x_test))

# metrics計算
from sklearn import metrics
print(metrics.classification_report(y_test, predicted_results))
y_probs = model.predict_proba(x_test)[:,0]


# 繪圖時間
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from itertools import cycle

plt.figure(1)
plt.plot([0,1], [0,1], "k--")
plt.xlabel("recall")
plt.ylabel("precision")
plt.title("Precision v.s. Recall curve")

colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])
saving = []
precision = dict()
recall = dict()

#進入迴圈，畫出13張圖，代表13個classifier
for k,color in zip(range(13), colors):

    # 產出label和not label的curves
    for copies in y_test:
        saving.append(copies)
    count = 0
    for nums in saving:
        if nums != k+1:
            saving[count] = 0
        count += 1

    # 繪圖
    precision[k], recall[k], _ = precision_recall_curve(y_test, predicted_results, pos_label = k+1)
    plt.plot(recall[k], precision[k], lw=2, label='class {}'.format(k+1))
    saving = []

# 誤導的這個Display(他只能畫一個)
# from sklearn.metrics import PrecisionRecallDisplay
# display.plot(name=f"Precision-recall for class {k}", color=color)
# display = PrecisionRecallDisplay(precision,recall)


plt.legend(loc='best')  # 調整位置
plt.savefig('Precision v.s. Recall curve.svg')

#-------

Id_list = []
import csv
file = open('D:/Coding/text_mining/hw2_sam.csv')
reader = csv.DictReader(file)
for row in reader:
    Id_list.append(row['Id'])
    if row['Id'] == str(17):
        testing_data = binary_vectors[16]
    else:
        testing_data = vstack((testing_data,binary_vectors[int(row['Id'])-1]))

predict_result2 = []
predict_result2.extend(model.predict(testing_data))

# 寫入CSV檔案
with open('Bernoulli.csv', 'w', newline='') as csvfile:
  # 以空白分隔欄位，建立 CSV 檔寫入器
    writer = csv.writer(csvfile)
    writer.writerow(['Id', 'Value'])
    for i in range(len(Id_list)):
        writer.writerow([Id_list[i],predict_result2[i]])