import streamlit as st
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from networkx.algorithms.community import greedy_modularity_communities
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="Social Network Analysis Tool", layout="wide")

st.title("Dynamic Social Network Analysis Tool")

# Sidebar input options
st.sidebar.header("Input Options")
data_source = st.sidebar.radio("Select Data Source", ["Preloaded Data", "Upload CSV", "Manual Input"])

if data_source == "Preloaded Data":
    data = {
        "Person_A": ["Alice", "Alice", "Bob", "Charlie", "David"],
        "Person_B": ["Bob", "Charlie", "Charlie", "David", "Eve"],
        "Connection_Strength": [3, 5, 2, 4, 1],
    }
    df = pd.DataFrame(data)

elif data_source == "Upload CSV":
    st.sidebar.write("**Expected CSV Format**")
    st.sidebar.markdown("""
    - The CSV file must contain **three columns**:
        1. `Person_A` (Name of the first person in the connection)
        2. `Person_B` (Name of the second person in the connection)
        3. `Connection_Strength` (Numerical value representing the strength of the connection)
    - Ensure column names match exactly, and data is properly formatted.
    - Example:
        ```csv
        Person_A,Person_B,Connection_Strength
        Alice,Bob,3
        Alice,Charlie,5
        Bob,Charlie,2
        Charlie,David,4
        David,Eve,1
        ```
    """)

    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            # Validate column names
            expected_columns = {"Person_A", "Person_B", "Connection_Strength"}
            if not expected_columns.issubset(df.columns):
                st.error(f"The uploaded CSV must contain columns: {', '.join(expected_columns)}")
                st.stop()
            # Validate data types
            if not pd.api.types.is_numeric_dtype(df["Connection_Strength"]):
                st.error("The `Connection_Strength` column must contain numeric values.")
                st.stop()
        except Exception as e:
            st.error(f"Error reading the file: {e}")
            st.stop()
    else:
        st.warning("Please upload a CSV file.")
        st.stop()


elif data_source == "Manual Input":
    st.sidebar.write("Enter data manually below:")
    manual_data = st.sidebar.text_area(
    "Enter rows in the format: `Person_A,Person_B,Connection_Strength`",
    value="Alice,Bob,3\nAlice,Charlie,5\nBob,Charlie,2\nCharlie,David,4\nDavid,Eve,1"
    )
    if manual_data:
        try:
            rows = [row.split(",") for row in manual_data.split("\n") if row.strip()]
            df = pd.DataFrame(rows, columns=["Person_A", "Person_B", "Connection_Strength"])
            df["Connection_Strength"] = df["Connection_Strength"].astype(int)
        except Exception as e:
            st.error("Error parsing manual input data. Ensure the format is correct.")
            st.stop()
    else:
        st.warning("Please input data.")
        st.stop()

# Display the data
st.subheader("Social Network Data")
st.dataframe(df)

# Create and analyze the graph
G = nx.Graph()
for _, row in df.iterrows():
    G.add_edge(row["Person_A"], row["Person_B"], weight=row["Connection_Strength"])

st.subheader("Graph Analysis")
st.write(f"Nodes in the network: {list(G.nodes)}")
st.write(f"Edges in the network: {list(G.edges(data=True))}")

degree_centrality = nx.degree_centrality(G)
st.write("Degree Centrality:", degree_centrality)

communities = list(greedy_modularity_communities(G))
st.write(f"Communities detected: {len(communities)}")
for i, community in enumerate(communities, 1):
    st.write(f"Community {i}: {community}")

# Visualize the graph
st.subheader("Graph Visualization")
layout = st.selectbox("Select Graph Layout", ["spring", "circular", "kamada_kawai"])
pos = getattr(nx, f"{layout}_layout")(G)

plt.figure(figsize=(10, 8))
nx.draw(
    G, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=10, font_weight="bold"
)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
st.pyplot(plt)

# Interactive visualization
st.subheader("Interactive Visualization")
try:
    edge_x, edge_y, node_x, node_y = [], [], [], []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    for node in G.nodes:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="gray"), hoverinfo="none"))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text", text=list(G.nodes),
        marker=dict(size=10, color="blue", line=dict(width=2)),
        textposition="top center"
    ))
    fig.update_layout(title="Interactive Social Network", showlegend=False)
    st.plotly_chart(fig)
except ImportError:
    st.warning("Plotly is not installed. Install it for advanced visualizations.")

# Highlight node functionality
st.sidebar.header("Highlight Node")
highlight_node = st.sidebar.text_input("Enter a node to highlight", "")
if highlight_node and highlight_node in G:
    neighbors = list(G.neighbors(highlight_node))
    st.sidebar.write(f"Neighbors of {highlight_node}: {neighbors}")
    nx.draw(
        G, pos, with_labels=True, node_size=2000,
        node_color=["green" if node == highlight_node else "skyblue" for node in G.nodes],
        font_size=10, font_weight="bold"
    )
    st.pyplot(plt)
