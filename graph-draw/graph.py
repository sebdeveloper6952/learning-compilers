import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt

NODE_SIZE = 500

# graph representation
gJson = {
    1: {2: 'a'},
    2: {3: 'b', 4: 'a'}
}

edge_labels = {}
graph = nx.DiGraph()
for node_from in gJson.keys():
    for node_to in gJson[node_from]:
        graph.add_edges_from([(node_from, node_to)])
        edge_labels[(node_from, node_to)] = gJson[node_from][node_to]

# graph drawing stuff
pos = nx.layout.spring_layout(graph)
nodes = nx.draw_networkx_nodes(
    graph, 
    pos, 
    node_size=NODE_SIZE,
    node_color="black",
)
edges = nx.draw_networkx_edges(
    graph,
    pos,
    node_size=NODE_SIZE,
    arrows=True,
    arrowstyle="-|>",
    arrowsize=12,
    edge_cmap=plt.cm.Blues
)
nx.draw_networkx_labels(
    graph,
    pos,
    font_size=20,
    font_color='white'
)
nx.draw_networkx_edge_labels(
    graph,
    pos,
    edge_labels=edge_labels,
    font_color='black',
)

ax = plt.gca()
ax.set_axis_off()
plt.show()