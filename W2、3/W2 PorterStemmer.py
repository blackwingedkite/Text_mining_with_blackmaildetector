from nltk.tokenize import word_tokenize  # 分割用
from nltk.stem.porter import PorterStemmer
from collections import Counter  # 記數用
stemmer = PorterStemmer()
source = """Kites have a long and varied history and many different types are flown individually and at festivals worldwide. Kites may be flown for recreation, art or other practical uses.  Sport kites can be flown in aerial ballet, sometimes as part of a competition.  Power kites are multi-line steerable kites designed to generate large forces  which can be used to power activities such as kite surfing, kite landboarding,  kite buggying and snow kiting"""
stop_punc = [".", ",", ":", ";"]
tokens_byWordtokenize = word_tokenize(source)  # 基礎分割
final_token = [stemmer.stem(str.lower(x)) for x in tokens_byWordtokenize if x not in stop_punc]  # 去除STOPWORDS
print(final_token)
bags_of_words = Counter(final_token)  # 計算次數
print(bags_of_words)