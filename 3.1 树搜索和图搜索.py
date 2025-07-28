import networkx as nx
import matplotlib.pyplot as plt
# 创建一个无向图
G = nx.Graph()
# 添加边，表示两个节点之间的通路
G.add_edge('A', 'B', weight=5)
G.add_edge('B', 'C', weight=5)
G.add_edge('A', 'D', weight=3)
G.add_edge('A', 'E', weight=6)
G.add_edge('D', 'F', weight=4)
G.add_edge('E', 'I', weight=7)
G.add_edge('E', 'H', weight=4)
G.add_edge('E', 'G', weight=3)
G.add_edge('F', 'G', weight=4)
G.add_edge('G', 'H', weight=3)
G.add_edge('G', 'K', weight=5)
G.add_edge('H', 'J', weight=7)
G.add_edge('I', 'J', weight=5)
G.add_edge('J', 'K', weight=3)
G.add_edge('K', 'L', weight=6)

# 使用搜索算法输出所有A到K的路径及其长度
# 设置cutoff参数，限制搜索深度，防止无限搜索（树搜索中可能存在环）
for path in nx.all_simple_paths(G, source='A', target='K'):
    print(path, nx.path_weight(G, path, 'weight'))  # 输出路径和路径长度

nx.draw(G, with_labels=True)
plt.show()
