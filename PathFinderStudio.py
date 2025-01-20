import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import random

# --- Helper Functions ---
def generate_random_graph(num_nodes, num_edges):
    G = nx.Graph()
    for i in range(num_nodes):
        G.add_node(i)
    for _ in range(num_edges):
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        weight = random.randint(1, 20)
        if u != v:
            G.add_edge(u, v, weight=weight)
    return G

def visualize_graph(G, pos, title, highlight_edges=None, highlight_nodes=None):
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=700, font_size=10)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    if highlight_edges:
        nx.draw_networkx_edges(G, pos, edgelist=highlight_edges, edge_color='red', width=2)
    if highlight_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=highlight_nodes, node_color='orange', node_size=700)
    plt.title(title)
    st.pyplot(plt)

def dijkstra(G, start):
    return nx.single_source_dijkstra_path_length(G, start)

def minimum_spanning_tree(G):
    return nx.minimum_spanning_tree(G, algorithm='kruskal')

def depth_first_search(G, start):
    return list(nx.dfs_edges(G, source=start))

def breadth_first_search(G, start):
    return list(nx.bfs_edges(G, source=start))

# --- Streamlit App ---
st.title("Graph Algorithm Visualizer")

if 'graph' not in st.session_state:
    st.session_state.graph = None

if 'graph_pos' not in st.session_state:
    st.session_state.graph_pos = None

# --- Sidebar Options ---
st.sidebar.title("Graph Options")
graph_type = st.sidebar.selectbox("Graph Type", ["Random Graph", "User Input Graph"])

if graph_type == "Random Graph":
    
    num_nodes = st.sidebar.slider("Number of Nodes", 5, 20, 10, help="For better graph structure ensure that number of nodes are less than the nu,ber of edges")
    num_edges = st.sidebar.slider("Number of Edges", 5, 40, 15)
    if st.sidebar.button("Generate Graph"):
        st.session_state.graph = generate_random_graph(num_nodes, num_edges)
        st.session_state.graph_pos = nx.spring_layout(st.session_state.graph)

elif graph_type == "User Input Graph":
    st.sidebar.write("Use the format 'node1 node2 weight' to define edges.")
    edges_input = st.sidebar.text_area("Enter edges (one per line):", "0 1 4\n1 2 3\n2 3 2")
    if st.sidebar.button("Generate Graph"):
        G = nx.Graph()
        for line in edges_input.strip().split("\n"):
            u, v, w = map(int, line.split())
            G.add_edge(u, v, weight=w)
        st.session_state.graph = G
        st.session_state.graph_pos = nx.spring_layout(G)

# --- Ensure Graph Exists ---
if st.session_state.graph is None:
    st.warning("Generate a graph to start visualizing.")
    st.stop()

G = st.session_state.graph
pos = st.session_state.graph_pos

# --- Algorithm Selection ---
st.sidebar.title("Algorithm Options")
algorithm = st.sidebar.selectbox("Select Algorithm", ["Dijkstra", "Minimum Spanning Tree", "DFS", "BFS"])
start_node = st.sidebar.selectbox("Select Start Node", list(G.nodes()))

# --- Display Original Graph ---
st.subheader("Original Graph")
visualize_graph(G, pos, "Original Graph")

# --- Algorithm Visualization ---
st.subheader(f"{algorithm} Visualization")
if algorithm == "Dijkstra":
    distances = dijkstra(G, start_node)
    st.write("Shortest Path Distances:", distances)
    visualize_graph(G, pos, f"{algorithm} Algorithm (Start: {start_node})", highlight_nodes=[start_node])

elif algorithm == "Minimum Spanning Tree":
    mst = minimum_spanning_tree(G)
    mst_edges = list(mst.edges(data=True))
    st.write("Minimum Spanning Tree Edges:", mst_edges)
    visualize_graph(G, pos, "Minimum Spanning Tree", highlight_edges=mst.edges())

elif algorithm == "DFS":
    dfs_edges = depth_first_search(G, start_node)
    st.write("DFS Traversal Edges:", dfs_edges)
    visualize_graph(G, pos, f"DFS Traversal (Start: {start_node})", highlight_edges=dfs_edges)

elif algorithm == "BFS":
    bfs_edges = breadth_first_search(G, start_node)
    st.write("BFS Traversal Edges:", bfs_edges)
    visualize_graph(G, pos, f"BFS Traversal (Start: {start_node})", highlight_edges=bfs_edges)
