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

# 使用Dijkstra算法找到从A到K的最短路径
path = nx.dijkstra_path(G, source='A', target='K')
length = nx.dijkstra_path_length(G, source='A', target='K')

print(path, length)
