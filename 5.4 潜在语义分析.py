from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 示例文档集合
documents = pd.DataFrame.from_dict({
    'a1': 'Efficient Algorithms for Non-convex Isotonic Regression through Submodular Optimization.',
    'a2': 'Combinatorial Optimization with Graph Convolutional Networks and Guided Tree Search.',
    'a3': 'An Improved Analysis of Alternating Minimization for Structured Multi-Response Regression.',
    'a4': 'Analysis of Krylov Subspace Solutions of Regularized Non-Convex Quadratic Problems.',
    'a5': 'Post: Device Placement with Cross-Entropy Minimization and Proximal Policy Optimization.',
    'b1': 'CRISPR/Cas9 and TALENs generate heritable mutations for genes involved in small RNA processing of Glycine max and Medicago truncatula.',
    'b2': 'Generation of D1-1 TALEN isogenic control cell line from Dravet syndrome patient iPSCs using TALEN-mediated editing of the SCN1A gene.',
    'b3': 'Genome-Scale CRISPR Screening Identifies Novel Human Pluripotent Gene Networks.',
    'b4': 'Champions: A phase 1/2 clinical trial with dose escalation of SB-913 ZFN-mediated in vivo human genome editing for treatment of MPSII (Hunter syndrome).',
}, orient='index', columns=['abstract'])

# 词频统计
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents['abstract'].values)
# 输出词表
print('词表索引：')
vocabolary = vectorizer.get_feature_names_out()
print(pd.DataFrame.from_dict(vectorizer.vocabulary_,
      orient='index', columns=['number']).sort_values(by='number'))
# 使用LSA进行主题分析
n_components = 5
lsa = TruncatedSVD(n_components=n_components)
lsa.fit(X)
# 输出LSA主题
print('LSA主题：')
topic_words = []
for i, comp in enumerate(lsa.components_):
    termsInComp = zip(vocabolary, comp)
    sortedTerms = sorted(termsInComp, key=lambda x: x[1], reverse=True)[:10]
    topic_words.append(sortedTerms)
    print("Topic %d:" % (i+1))
    print(sortedTerms)

# 绘制相关系数矩阵
fig, ax = plt.subplots(2, 2, figsize=(8, 8))
# 计算文档-文档相关系数矩阵
A1 = np.abs(np.corrcoef(X.toarray()))
sns.heatmap(A1, ax=ax[0, 0], vmin=0, vmax=1)
ax[0, 0].set_xticks(range(len(documents)))
ax[0, 0].set_xticklabels(documents.index)
ax[0, 0].set_yticks(range(len(documents)))
ax[0, 0].set_yticklabels(documents.index)
ax[0, 0].set_title('转换前的文档-文档相关系数矩阵')
A2 = np.abs(np.corrcoef(lsa.fit_transform(X)))
sns.heatmap(A2, ax=ax[0, 1], vmin=0, vmax=1)
ax[0, 1].set_title('转换后的文档-文档相关系数矩阵')
ax[0, 1].set_xticks(range(len(documents)))
ax[0, 1].set_xticklabels(documents.index)
ax[0, 1].set_yticks(range(len(documents)))
ax[0, 1].set_yticklabels(documents.index)
# 计算单词-单词相关系数矩阵
B1 = np.abs(np.corrcoef(X.toarray().T))
sns.heatmap(B1, ax=ax[1, 0], vmin=0, vmax=1)
ax[1, 0].axis('off')
ax[1, 0].set_title('转换前的单词-单词相关系数矩阵')
B2 = np.abs(np.corrcoef(lsa.components_))
sns.heatmap(B2, ax=ax[1, 1], vmin=0, vmax=1)
ax[1, 1].axis('off')
ax[1, 1].set_title('转换后的单词-单词相关系数矩阵')
plt.tight_layout()
plt.show()
