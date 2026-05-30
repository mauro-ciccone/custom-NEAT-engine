import json
import networkx as nx
import matplotlib.pyplot as plt

def visualize_champion(filepath="xor_trained_backup.json"):
    # 1. Load the frozen data
    with open(filepath, "r") as f:
        data = json.load(f)

    # 2. Find the smartest brain in the file
    genomes = data["genomes"]
    champion = max(genomes, key=lambda g: g["fitness"])
    print(f"Drawing Champion with Fitness: {champion['fitness']:.3f}")

    # 3. Create a blank canvas
    G = nx.DiGraph()

    # 4. Draw the physical Neurons (Nodes)
    for node in champion["neurons"]:
        # Assign colors and layers based on the type of node
        if node["type"] in ["input", "bias"]:
            layer = 0
            color = "lightgreen"
        elif node["type"] == "output":
            layer = 2
            color = "salmon"
        else:
            layer = 1
            color = "lightblue"
            
        G.add_node(node["id"], layer=layer, color=color)

    # 5. Draw the Cables (Synapses)
    for syn in champion["synapses"]:
        # We only draw cables that evolution hasn't disabled!
        if syn["enabled"]:
            # Make heavy weights thicker
            thickness = min(abs(syn["weight"]), 5.0) 
            # Blue for positive signals, Red for negative (inhibitory) signals
            edge_color = "blue" if syn["weight"] > 0 else "red" 
            
            G.add_edge(syn["in"], syn["out"], weight=thickness, color=edge_color, actual=syn["weight"])

    # 6. Organize it so Inputs are on the left, Outputs on the right
    pos = nx.multipartite_layout(G, subset_key="layer")
    
    node_colors = [data["color"] for _, data in G.nodes(data=True)]
    edge_colors = [data["color"] for _, _, data in G.edges(data=True)]
    edge_widths = [data["weight"] for _, _, data in G.edges(data=True)]

    # 7. Render the picture!
    plt.figure(figsize=(10, 6))
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=node_colors,
        edge_color=edge_colors,
        width=edge_widths,
        node_size=800,
        font_size=10,
        font_weight="bold",
        arrowsize=20
    )

    edge_labels = {(u, v): f"{d['actual']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=8,
        font_color="black",
        label_pos=0.3 # Offsets the text slightly so it doesn't pile up in the middle
    )
    
    plt.title(f"NEAT Champion Brain (Fitness: {champion['fitness']:.3f})")
    plt.show()

if __name__ == "__main__":
    visualize_champion()