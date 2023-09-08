final = []
with open(file = 'D:/Coding/text_mining/training_new.txt',mode = 'r') as f:
    content = f.read()
    list1 = content.split("\n")
    list2 = []
    
    for it in list1:
        list2.append(it.split())
    for i in list2:
        print(i)
        temp = []
        for index, label in enumerate(i):
            if index != 0:
                temp.append(int(label))
        final.append(temp)
train_label = []
train = []
test = []
for i in range(1,1096):
    append = False
    for j in final:
        for k in j:
            if i==k:
                train.append(i)
                label = final.index(j)+1
                append = True
                break
    if append == False:
        test.append(i)
print(train)
print(test)


# 1. Extract the [CLS] embedding of each training document
# Use the BERT-BASE pre-trained model

from keras_bert import extract_embeddings
from keras_bert import load_vocabulary
from keras_bert import Tokenizer
model_path = 'uncased_L-12_H-768_A-12'
dict_path = 'uncased_L-12_H-768_A-12/vocab.txt'  # 在這裡就會出現第一個可略過的GPU無法搜尋問題

bert_token_dict = load_vocabulary(dict_path)
bert_tokenizer = Tokenizer(bert_token_dict)
cls_list = []
nums_list = []

# Put training data into a list (15*13)
index_list = []
with open(file = 'D:/Coding/text_mining/training_new.txt',mode = 'r') as f:
    for i in range(1,14):
        nums = f.readline().split(" ")
        for j in nums[1:-1]:
            index_list.append(j)
            nums_list.append(i)

#Using the embeddings to construct a SVM classification model(Linear kernel)
#目標：(195, 768)的 <class 'scipy.sparse._csr.csr_matrix'>

from scipy import sparse
from scipy.sparse import vstack
for k in index_list[0]:
    with open(file = 'D:/Coding/text_mining/PA1-data/'+str(k)+'.txt',mode = 'r') as f:
        single_word = [f.read()]
        embeddings = extract_embeddings(model_path, single_word) # <class 'numpy.ndarray'>
        sparce_embeddings = sparse.csr_matrix(embeddings[0][0])
        print("the first one")
   
for k in index_list[1:-1]:
    with open(file = 'D:/Coding/text_mining/PA1-data/'+str(k)+'.txt',mode = 'r') as f:
        single_word = [f.read()]
        embeddings = extract_embeddings(model_path, single_word) # <class 'numpy.ndarray'>
        sparce_embeddings = vstack(sparce_embeddings,sparse.csr_matrix(embeddings[0][0]))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(sparce_embeddings, nums_list, test_size=0.1, random_state=3, stratify = nums_list)

from sklearn.svm import SVC
SVM_model = SVC(kernel='linear', C=1.0)
SVM_model.fit(x_train,y_train)
predicted_results = []
expected_results = []
expected_results.extend(y_test)
predicted_results.extend(SVM_model.predict(x_test))
print(predicted_results)

from sklearn import metrics
print(metrics.classification_report(expected_results, predicted_results))

Id_list = []
import csv
file = open('D:/Coding/text_mining/testing.csv')
reader = csv.DictReader(file)
for row in reader:
    Id_list.append(row['Id'])
    if row['Id'] == str(17):
        with open(file = 'D:/Coding/text_mining/PA1-data/'+int(row['Id'])+'.txt',mode = 'r') as f:
            single_word = [f.read()]
            embeddings = extract_embeddings(model_path, single_word)
            testing_data = sparse.csr_matrix(embeddings[0][0])
    else:
        with open(file = 'D:/Coding/text_mining/PA1-data/'+int(row['Id'])+'.txt',mode = 'r') as f:
            single_word = [f.read()]
            embeddings = extract_embeddings(model_path, single_word)
            testing_data = vstack(testing_data,sparse.csr_matrix(embeddings[0][0]))

predict_result2 = []
predict_result2.extend(SVM_model.predict(testing_data))

# 寫入CSV檔案
with open('BERT_estimation.csv', 'w', newline='') as csvfile:
  # 以空白分隔欄位，建立 CSV 檔寫入器
    writer = csv.writer(csvfile)
    writer.writerow(['Id', 'Value'])
    for i in range(len(Id_list)):
        writer.writerow([Id_list[i],predict_result2[i]])