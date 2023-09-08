# 1. Extract the [CLS] embedding of each training document
# Use the BERT-BASE pre-trained model
from keras_bert import extract_embeddings
from keras_bert import POOL_NSP
model_path = 'uncased_L-12_H-768_A-12'
dict_path = 'uncased_L-12_H-768_A-12/vocab.txt'

# Put training data into a list (15*13)
paired_nums = []
index_list = []
with open(file = './training.txt',mode = 'r') as f:
    for i in range(1,14):
        nums = f.readline().split(" ")
        for j in nums[1:-1]:
            index_list.append(int(j))
            paired_nums.append(int(i))
# Using the embeddings to construct a SVM classification model(Linear kernel)
word_embeddings = []
for k in index_list:
    print(k)
    with open(file = './PA1-data/'+str(k)+'.txt',mode = 'r') as f:
        single_word = [f.read()]
    embeddings = extract_embeddings(model_path, single_word, poolings=[POOL_NSP])
    word_embeddings.append(embeddings[0])

from sklearn.svm import SVC
SVM_model = SVC(kernel='linear', C=1)
SVM_model.fit(word_embeddings,paired_nums)

ID_list = []
test_list = []
for i in range(1,1096):
    if i not in index_list:
        print(i)
        ID_list.append(i)
        with open(file = './PA1-data/'+str(i)+'.txt',mode = 'r') as f:
            single_word = [f.read()]
            embeddings = extract_embeddings(model_path, single_word, poolings=[POOL_NSP])
            test_list.append(embeddings[0])
predict_result2 = []
predict_result2.extend(SVM_model.predict(test_list))

# 寫入CSV檔案
import csv
with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Id', 'Value'])
    for i in range(len(ID_list)):
        writer.writerow([ID_list[i],predict_result2[i]])