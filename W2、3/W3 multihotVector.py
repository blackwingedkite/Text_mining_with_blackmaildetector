from nltk.tokenize import word_tokenize  # 分割用
from collections import Counter
source = ["Kites have a long and varied history and many different types are flown individually and at festivals worldwide", "Kites may be flown for recreation, art or other practical uses.", "Sport kites can be flown in aerial ballet, sometimes as part of a competition.", "Power kites are multi-line steerable kites designed to generate large forces which can be used to power activities such as kite surfing, kite landboarding,  kite buggying and snow kiting"]
stop_punc = [".", ",", ":", ";"]
each_docs_tokens = []

for doc in source:
    tokens_byWordtokenize = word_tokenize(doc)  # 基礎分割
    final_token = [x for x in tokens_byWordtokenize if x not in stop_punc]
    each_docs_tokens += [final_token]

all_tokens = []
for x in each_docs_tokens:  # x = 被斷過的單句
    all_tokens.extend(x)  # extend = 把被斷過的單句(list型態)加入all_tokens裡面

vocab = sorted(set(all_tokens))  # 去除重複項
multihot_vector = [0]*len(vocab)
bag_of_words = Counter(vocab)
for key, values in bag_of_words.items():
    multihot_vector[vocab.index(key)] = 1

print(multihot_vector) 