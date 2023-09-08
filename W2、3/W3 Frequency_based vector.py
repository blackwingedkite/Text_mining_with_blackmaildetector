from nltk.tokenize import word_tokenize  # 分割用
from nltk.stem.porter import PorterStemmer
import numpy as np
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

TF_vector = np.zeros((len(source), len(vocab)), float)  # 4個長長的vector
for i, token_list in enumerate(each_docs_tokens):
    token_Count = Counter(token_list)  # 把4個list裡面的東西算起來
    for key, values in token_Count.items():  # Counter是dictionary型態
        TF_vector[i][vocab.index(key)] = values / len(token_list)  # 總長度

print(TF_vector)  # 最後計算出每個字詞的佔比

  # 如果是multiple_hot vector的話，只需要把最後的值改成1就好，我耍白癡了