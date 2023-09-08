from keras_bert import extract_embeddings
from keras_bert import load_vocabulary
from keras_bert import Tokenizer
model_path = 'uncased_L-12_H-768_A-12'
dict_path = 'uncased_L-12_H-768_A-12/vocab.txt'

# bert_token_dict = load_vocabulary(dict_path)
# bert_tokenizer = Tokenizer(bert_token_dict)
single_input_text = ['all work and no play.']
# tokens = bert_tokenizer.tokenize(single_input_text[0])  # ['[CLS]', 'all', 'work', 'and', 'no', 'play', '.', '[SEP]']
# print(bert_tokenizer.tokenize(single_input_text))  # ERROR
# indices, segments = bert_tokenizer.encode(single_input_text[0])
# print(indices)  # [101, 2035, 2147, 1998, 2053, 2377, 1012, 102]
# print(segments)  # [0, 0, 0, 0, 0, 0, 0, 0]

# embeddings = extract_embeddings(model_path, single_input_text)
# print(len(embeddings),len(embeddings[0]),len(embeddings[0][0]))  # [1,8,768]
# print(embeddings[0][0]) # 768維向量

from keras_bert import POOL_NSP
embeddings2 = extract_embeddings(model_path, single_input_text, poolings=[POOL_NSP])
print(len(embeddings2)) #1
print(embeddings2)  # 和[CLS]一樣的向量?
print(embeddings2[0])
print(len(embeddings2))
print(len(embeddings2[0]))
print(type(embeddings2))
print(type(embeddings2[0]))
print(type(embeddings2[0][0]))
print(embeddings2[0][0])
# print(type(embeddings))  # list
# print(type(embeddings[0]))  # ndarray
# print(type(embeddings[0][0]))  # ndarray
# print(type(embeddings2))  # list

# two_input_text = [('all work and no play.', 'this is an order.')]

# two_inputs_embeddings = extract_embeddings(model_path, two_input_text)
# tokens1 = bert_tokenizer.tokenize(two_input_text[0][0])
# tokens2 = bert_tokenizer.tokenize(two_input_text[0][1])
# print(tokens1,tokens2)
# print(len(two_inputs_embeddings[0]))  # 14
# print(two_inputs_embeddings[0][0])  # 一樣是768維向量

# tokens = bert_tokenizer.tokenize(first='all work and no play.', second='this is an order.')
# indices, segments = bert_tokenizer.encode(first='all work and no play.', second='this is an order.')
# print(len(tokens))
# print(indices)
# print(segments)
# print(tokens)