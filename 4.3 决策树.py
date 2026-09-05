from sklearn import tree
import pandas as pd
import plotly.express as px
from skimage.io import imread
import graphviz
import os
data = pd.DataFrame({
    '年龄': ['>30', '>30', '20-30', '<20', '<20', '<20', '20-30', '>30', '>30', '<20', '>30', '20-30', '20-30', '<20'],
    '银行流水': ['高', '高', '高', '中', '低', '低', '低', '中', '低', '中', '中', '中', '高', '中'],
    '是否结婚': ['否', '否', '否', '否', '否', '是', '是', '否', '是', '否', '是', '否', '是', '否'],
    '拥有房产': ['是', '否', '是', '是', '是', '否', '否', '是', '是', '是', '否', '否', '是', '否'],
    '是否给予贷款': ['否', '否', '是', '是', '是', '否', '是', '否', '是', '是', '是', '是', '是', '否']
})

# 将数据集转换为数字
data['年龄'] = data['年龄'].map({'>30': 30, '20-30': 20, '<20': 10})
data['银行流水'] = data['银行流水'].map({'高': 3, '中': 2, '低': 1})
data['是否结婚'] = data['是否结婚'].map({'是': 1, '否': 0})
data['拥有房产'] = data['拥有房产'].map({'是': 1, '否': 0})
data['是否给予贷款'] = data['是否给予贷款'].map({'是': 1, '否': 0})

# 生成决策树
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(data.iloc[:, 0:4], data.iloc[:, 4])

# 可视化决策树
# 使用graphviz可视化决策树
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=data.columns[0:4],
                                class_names=['否', '是'],
                                filled=True, rounded=True,
                                special_characters=True,
                                fontname='SimHei')
graph = graphviz.Source(dot_data)
graph.render('决策树', format='png')  # 保存决策树
fig = px.imshow(imread('决策树.png'))
os.remove('决策树')
os.remove('决策树.png')
fig.show()
