from keras_bert import extract_embeddings
from keras_bert import load_vocabulary
from keras_bert import Tokenizer
model_path = 'uncased_L-12_H-768_A-12'
dict_path = 'uncased_L-12_H-768_A-12/vocab.txt'

bert_token_dict = load_vocabulary(dict_path)
bert_tokenizer = Tokenizer(bert_token_dict)

with open(file = 'D:/coding/text_mining/PA1-data/1.txt',mode = 'r') as f:
    single_input_text = [f.read()]
tokens = bert_tokenizer.tokenize(single_input_text[0]) 
print(tokens)

embeddings = extract_embeddings(model_path, single_input_text)
print(len(embeddings),len(embeddings[0]),len(embeddings[0][0]))  # [1,379,768]
print(embeddings[0][0]) # 768維向量