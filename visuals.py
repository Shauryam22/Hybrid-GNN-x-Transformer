import networkx as nx
import matplotlib.pyplot as plt
from build_co_occurence import build_coocc
from load import preproc_data
from fusion_model import HumanAIDetector
import torch
import torch.nn.functional as F
from metrics import eval_fusion_ablation,print_full_metrics
device = 'cuda'
block_size = 128
batch_size = 64
vocab_size,itos,train_dataset,val_dataset,train_loader,val_loader = preproc_data(block_size,batch_size,device)

adj_dense = build_coocc(train_dataset,vocab_size=vocab_size,window=5)
indices=(adj_dense>0).nonzero(as_tuple = False).t()  # storing the elements index as tuple hence (_,2)
indices.shape
values = adj_dense[indices[0],indices[1]]
adj_sparse = torch.sparse_coo_tensor(indices,values,
                                     size=adj_dense.shape).coalesce().to(device)

def plot_word_graph(adj_sparse, itos, vocab_size, top_k_edges=100):
    # 1. Extract edges
    coo_matrix = adj_sparse.coalesce()
    indices = coo_matrix.indices().cpu().numpy().T
    values = coo_matrix.values().cpu().numpy()

    # Create Graph
    G = nx.Graph()

    # 2. Filter and Add Edges
    # Combine indices and values to sort them
    edges_data = []
    for i in range(len(indices)):
        u, v = indices[i]
        w = values[i]
        # Ignore self-loops and 0 weights
        if u != v and w > 0: 
            edges_data.append((u, v, w))
    
    # Sort by weight (highest first) and keep only top_k to prevent clutter
    edges_data.sort(key=lambda x: x[2], reverse=True)
    edges_data = edges_data[:top_k_edges]

    # Add these filtered edges to Graph
    for u, v, w in edges_data:
        # Scale weight for visibility (since normalized values are small)
        G.add_edge(u, v, weight=w)

    # Only add nodes that are part of the top edges (to avoid empty floating dots)
    active_nodes = set()
    for u, v, w in edges_data:
        active_nodes.add(u)
        active_nodes.add(v)
        
    # 3. Layout and Draw
    pos = nx.spring_layout(G, k=0.5, seed=42) 
    
    # Get weights for drawing width
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    # Normalize widths for display (e.g., multiply by 10 so lines are visible)
    width = [w * 10 for w in weights] 

    plt.figure(figsize=(12, 12))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=list(active_nodes), 
                           node_size=100, node_color='skyblue', alpha=0.9)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=width, alpha=0.5, edge_color='gray')
    
    # Draw labels
    # Create a label dict for only the active nodes
    labels = {i: itos[i] for i in active_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
    
    plt.title(f"Top {top_k_edges} Strongest Word Connections", fontsize=15)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Run it
plot_word_graph(adj_sparse, itos, vocab_size, top_k_edges=500)

def visualize_top_nodes_subgraph(adj_sparse, itos, top_nodes=100):

    adj_cpu = adj_sparse.cpu().coalesce()
    indices = adj_cpu.indices()
    values = adj_cpu.values()

    vocab_size = len(itos)

    # Compute weighted degree for each node
    degree = torch.zeros(vocab_size)

    for i in range(indices.shape[1]):
        u = indices[0, i]
        v = indices[1, i]
        w = values[i]
        degree[u] += w
        degree[v] += w

    # Get top nodes by weighted degree
    top_ids = torch.topk(degree, top_nodes).indices.tolist()
    top_set = set(top_ids)

    G = nx.Graph()

    # Add edges only if both nodes are in top set
    for i in range(indices.shape[1]):
        u = indices[0, i].item()
        v = indices[1, i].item()
        w = values[i].item()

        if u in top_set and v in top_set and u < v:
            G.add_edge(itos[u], itos[v], weight=w)

    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(G, pos, node_size=1200, node_color='skyblue')
    nx.draw_networkx_labels(G, pos, font_size=9)

    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights)
    edge_widths = [w / max_weight * 5 for w in edge_weights]

    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5)

    plt.title("Structural Backbone: Top 30 Tokens by Weighted Degree")
    plt.axis("off")
    plt.show()
visualize_top_nodes_subgraph(adj_sparse,itos)