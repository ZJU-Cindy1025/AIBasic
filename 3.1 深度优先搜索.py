import networkx as nx

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

# 使用深度优先搜索算法算法从A开始遍历图
path = nx.depth_first_search.dfs_successors(G, source='A')
print(list(path))  # 输出路径
