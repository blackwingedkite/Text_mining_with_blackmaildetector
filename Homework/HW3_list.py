# 1. Extract the [CLS] embedding of each training document
# Use the BERT-BASE pre-trained model
from keras_bert import extract_embeddings
from keras_bert import POOL_NSP
model_path = 'uncased_L-12_H-768_A-12'
dict_path = 'uncased_L-12_H-768_A-12/vocab.txt'

# Put training data into a list (15*13)
paired_nums = []
index_list = []
with open(file = 'D:/Coding/text_mining/training_new.txt',mode = 'r') as f:
    for i in range(1,14):
        nums = f.readline().split(" ")
        for j in nums[1:-1]:
            index_list.append(j)
            paired_nums.append(i)
print(paired_nums)
print(index_list)

#Using the embeddings to construct a SVM classification model(Linear kernel)
word_embeddings = []
for k in index_list:
    with open(file = 'D:/Coding/text_mining/PA1-data/'+str(k)+'.txt',mode = 'r') as f:
        single_word = [f.read()]
        embeddings = extract_embeddings(model_path, single_word, poolings=[POOL_NSP])
        word_embeddings.append(embeddings[0])

from sklearn.svm import SVC
SVM_model = SVC(kernel='rbf', gamma='scale', C=1.0)
SVM_model.fit(word_embeddings,paired_nums)

Id_list = []
import csv
file = open('D:/Coding/text_mining/hw2_sam.csv')
reader = csv.DictReader(file)
csvdata = []
for row in reader:
    Id_list.append(row['Id'])
    print(row['Id'])
    with open(file = 'D:/Coding/text_mining/PA1-data/'+str(row['Id'])+'.txt',mode = 'r') as f:
        single_word = [f.read()]
        embeddings = extract_embeddings(model_path, single_word, poolings=[POOL_NSP])
        csvdata.append(embeddings[0])
predict_result2 = []
predict_result2.extend(SVM_model.predict(csvdata))

# 寫入CSV檔案
with open('BERT_estimation_RBF.csv', 'w', newline='') as csvfile:
  # 以空白分隔欄位，建立 CSV 檔寫入器
    writer = csv.writer(csvfile)
    writer.writerow(['Id', 'Value'])
    for i in range(len(Id_list)):
        writer.writerow([Id_list[i],predict_result2[i]])