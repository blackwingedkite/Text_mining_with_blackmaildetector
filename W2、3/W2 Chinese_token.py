import jieba
sentence = "國立臺灣大學是一所積極新創、學科齊全、學術實力雄厚、辦學特色鮮明，在國際上具有重要影響力與競爭力的綜合性大學，在多個學術領域具有非常前瞻的科技實力，擁有世界一流的實驗室與師資力量，各種排名均位於全球前列。歡迎大家報考國立臺灣大學。"
jieba.set_dictionary("D:\Coding\dict.txt.big.txt")
new_result = jieba.cut(sentence)
new_tokens = list(new_result)
print(new_tokens)