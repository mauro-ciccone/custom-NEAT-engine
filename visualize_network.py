import json
import networkx as nx
import matplotlib.pyplot as plt

def visualize_champion(filepath):
    # 1. Load the frozen data
    with open(filepath, "r") as f:
        data = json.load(f)

    # 2. Find the smartest brain in the file
    genomes = data["genomes"]
    champion = max(genomes, key=lambda g: g["fitness"])
    print(f"Drawing Champion with Fitness: {champion['fitness']:.3f}")

    # 3. Create a blank canvas
    G = nx.DiGraph()

    # 4. Add all Nodes to the Graph first
    for node in champion["neurons"]:
        if node["type"] in ["input", "bias"]:
            color = "lightgreen"
        elif node["type"] == "output":
            color = "salmon"
        else:
            color = "lightblue"
        
        G.add_node(node["id"], type=node["type"], color=color)

    # 5. Add the Cables (Synapses)
    for syn in champion["synapses"]:
        if syn["enabled"]:
            thickness = min(abs(syn["weight"]), 5.0) 
            edge_color = "blue" if syn["weight"] > 0 else "red" 
            G.add_edge(syn["in"], syn["out"], weight=thickness, color=edge_color, actual=syn["weight"])

    # --- DYNAMIC LAYER CALCULATOR ---
    inputs = [n for n, d in G.nodes(data=True) if d['type'] in ['input', 'bias']]
    outputs = [n for n, d in G.nodes(data=True) if d['type'] == 'output']
    hidden = [n for n, d in G.nodes(data=True) if d['type'] not in ['input', 'bias', 'output']]

    for n in inputs:
        G.nodes[n]['layer'] = 0

    for n in hidden:
        max_dist = 1
        for i in inputs:
            try:
                dist = nx.shortest_path_length(G, source=i, target=n)
                if dist > max_dist:
                    max_dist = dist
            except nx.NetworkXNoPath:
                continue
        G.nodes[n]['layer'] = max_dist

    max_layer = max([G.nodes[n].get('layer', 0) for n in G.nodes()] + [0])
    for n in outputs:
        G.nodes[n]['layer'] = max_layer + 1

    # 6. Build the base layout
    pos = nx.multipartite_layout(G, subset_key="layer", align="vertical")
    
    # --- THE ARCHITECTURE SPLIT & VERTICAL STRETCH ---
    for n in G.nodes():
        x, y = pos[n]
        if G.nodes[n]['type'] in ['input', 'bias', 'output']:
            # Spread slightly (y * 2) and push down (-3)
            pos[n] = (x, (y * 2.0) - 3.0) 
        else:
            # Stretch MASSIVELY (y * 6) and push up (+3)
            pos[n] = (x, (y * 6.0) + 3.0) 
    
    node_colors = [data["color"] for _, data in G.nodes(data=True)]
    edge_colors = [data["color"] for _, _, data in G.edges(data=True)]
    edge_widths = [data["weight"] for _, _, data in G.edges(data=True)]

    # 7. Render the picture!
    plt.figure(figsize=(16, 10)) # Made the canvas slightly taller
    
    # Notice node_size is reduced from 800 to 500 to prevent crowding
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, edgecolors="black", alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")

    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_colors,
        width=edge_widths,
        arrowsize=15,
        connectionstyle="arc3,rad=0.15", 
        alpha=0.8
    )

    edge_labels = {(u, v): f"{d['actual']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=7,
        font_color="black",
        label_pos=0.3,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.5) 
    )
    
    plt.title(f"NEAT Champion Brain (Fitness: {champion['fitness']:.3f})")
    plt.axis("off") 
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    with open("config.json", "r") as file:
        config = json.load(file)

    visualize_champion(f"{config['environment']}_training_backup.json")