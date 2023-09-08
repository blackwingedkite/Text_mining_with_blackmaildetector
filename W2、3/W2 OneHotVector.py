from nltk.tokenize import word_tokenize  # 分割用
from nltk.stem.porter import PorterStemmer
import numpy as np
source = """Kites have a long and varied history and many different types are flown individually and at festivals worldwide. Kites may be flown for recreation, art or other practical uses.  Sport kites can be flown in aerial ballet, sometimes as part of a competition.  Power kites are multi-line steerable kites designed to generate large forces  which can be used to power activities such as kite surfing, kite landboarding,  kite buggying and snow kiting"""
stop_punc = [".", ",", ":", ";"]

tokens_byWordtokenize = word_tokenize(source)  # 基礎分割
final_token = [x for x in tokens_byWordtokenize if x not in stop_punc]
vocab = sorted(set(final_token))
print(vocab)

num_token = len(final_token)  # 72個，代表這串文字裡面總共有幾個字
vocab_size = len(vocab)  # 55個，字典裏面總共有55個字，代表有17個字是重複的
onehot_vectors = np.zeros((num_token, vocab_size), int)
for i, word in enumerate(final_token):  # enumerate: 配對用，給予編號
    onehot_vectors[i, vocab.index(word)] = 1 # 第一行，第vocab.index列，而字典的順序是被enumerate決定的
print(onehot_vectors)