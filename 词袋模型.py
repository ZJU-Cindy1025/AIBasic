import math

txt1 = "词袋模型是自然语言处理和信息检索中的一种常用文本表示方法。它将文本表示为一个词的集合，忽略词语的顺序和语法结构，只关注词语的出现频率或其他统计量"
txt2 = "词袋模型是一种将文本数据转换为数值型向量的方法，其中文本被视为一个由词语组成的无序集合，每个词语的出现都被独立考虑，而不考虑其上下文或顺序"

# 按字分词
set1 = set(txt1)
set2 = set(txt2)

# 构建词汇表
vocab = set1 | set2
vocab = list(vocab)
print("词汇表：", vocab)

# 计算词频
N = len(vocab)
fre1 = [0 for _ in range(N)]  # 句子1的词频向量
fre2 = [0 for _ in range(N)]  # 句子2的词频向量

for i in range(N):
    count = txt1.count(vocab[i])  # 计算某个字在句子1中出现的次数
    fre1[i] = count

    count = txt2.count(vocab[i])  # 计算某个字在句子2中出现的次数
    fre2[i] = count

# 计算余弦相似度
A = 0  # 2个向量的乘积
mod1 = 0  # 向量1的模
mod2 = 0  # 向量1的模
for i in range(N):
    A = A+fre1[i]*fre2[i]
    mod1 = mod1+fre1[i]*fre1[i]
    mod2 = mod2+fre2[i]*fre2[i]
mod1 = math.sqrt(mod1)
mod2 = math.sqrt(mod2)
cos = A/(mod1*mod2)  # 计算余弦相似度

print("句子1的词频向量：", fre1)
print("句子2的词频向量：", fre2)
print("余弦相似度", cos)
