import torch
import torch.nn as nn
import torch.optim as optim
import jieba
from tqdm import tqdm
import plotly.express as px

# 示例数据
corpus = """谋曹操
【奸雄】：游戏开始时，你可以选择获得至多两枚“治世”标记。当你受到伤害后，你可以获得对你造成伤害的牌，若你没有“治世”标记，你摸一张牌，然后你可以移除1枚“治世”标记。
【清正】：出牌阶段开始时，你可以弃置手牌中3-X种花色的所有牌，观看一名其他角色的手牌并弃置其中一种花色的所有牌，若弃置该角色的牌少于弃置你的牌，你对其造成1点伤害。然后若你有技能【奸雄】且“治世”标记的数量小于2，你可以获得1枚“治世”标记。（X为“治世”标记的数量且至多为2）
【护驾】：主公技，每轮限一次，当你受到伤害时，你可以将此伤害转移给一名魏势力其他角色。
"""

# 分词
tokens = [token for token in jieba.cut(corpus)]
# 清除无意义的停用词汇
with open('stop_words_full.txt', 'r', encoding='utf-8') as stopwords:
    for line in stopwords.readlines():
        while (line.strip('\n') in tokens):
            tokens.remove(line.strip('\n'))
# 构建词汇表
vocab = set(tokens)
vocab_size = len(vocab)
word_to_index = {word: i for i, word in enumerate(vocab)}
index_to_word = {i: word for word, i in word_to_index.items()}

# 构建训练数据
window_size = 2
training_data = []
for i in range(len(tokens)):
    target = tokens[i]
    context = [tokens[j] for j in range(
        max(0, i - window_size), min(len(tokens), i + window_size + 1)) if j != i]
    for word in context:
        training_data.append((target, word))
# 数据转换
targets = torch.tensor([word_to_index[target]
                       for target, _ in training_data], dtype=torch.long)
contexts = torch.tensor([word_to_index[context]
                        for _, context in training_data], dtype=torch.long)

# 词向量模型


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out_embeddings.weight = self.in_embeddings.weight

    def forward(self, target, context):
        in_embeds = self.in_embeddings(target)
        out_embeds = self.out_embeddings(context)
        scores = torch.matmul(in_embeds, out_embeds.t())
        return scores


# 使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
targets = targets.to(device)
contexts = contexts.to(device)
embedding_dim = 10
model = Word2Vec(vocab_size, embedding_dim)
model = model.to(device)
losses = []
# 训练
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1)
for epoch in tqdm(range(1000)):
    model.train()
    optimizer.zero_grad()
    scores = model(targets, contexts)
    loss = criterion(scores, contexts)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# 提取词向量
with torch.no_grad():
    word_vectors = model.in_embeddings.weight.cpu().numpy()

# 输出词向量
for word, index in word_to_index.items():
    print(f'Word: {word}, Vector: {word_vectors[index]}')

fig = px.line(y=losses, labels={'x': '迭代次数', 'y': '损失'})
fig.show()
