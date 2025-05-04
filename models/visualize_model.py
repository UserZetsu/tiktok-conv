import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import tempfile
import netron
import datetime
from tensorboard.plugins import projector
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import json

# Load the model
model = load_model('pklfiles/nn_model.h5')

# 1. Netron Visualization
def visualize_with_netron():
    # Save model to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
    model.save(temp_file.name)
    
    # Start Netron server
    netron.start(temp_file.name)
    print("Netron visualization started at http://localhost:8080")
    print("In the Netron interface, you can:")
    print("1. Click the 'Layout' button in the top toolbar")
    print("2. Select 'Horizontal' from the dropdown menu")
    print("3. Adjust the spacing using the 'Node Spacing' and 'Rank Spacing' sliders")
    print("Press Ctrl+C to stop the server")

# 2. TensorBoard Visualization
def visualize_with_tensorboard():
    # Create log directory
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create TensorBoard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True
    )
    
    # Create dummy data for visualization
    dummy_data = np.random.random((1, 66))  # 66 features as per the model input
    
    # Run model with dummy data to generate logs
    model.predict(dummy_data)
    
    print(f"TensorBoard logs saved to {log_dir}")
    print("Run 'tensorboard --logdir logs/fit' to view the visualization")

# 3. Custom NetworkX Visualization
def visualize_with_networkx():
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes for each layer
    layer_names = [layer.name for layer in model.layers]
    for i, name in enumerate(layer_names):
        G.add_node(name, layer_type=type(model.layers[i]).__name__)
    
    # Add edges between layers
    for i in range(len(layer_names)-1):
        G.add_edge(layer_names[i], layer_names[i+1])
    
    # Draw the graph
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, arrows=True)
    
    # Draw labels
    labels = {node: f"{node}\n({G.nodes[node]['layer_type']})" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels)
    
    plt.title("Neural Network Architecture")
    plt.axis('off')
    plt.savefig('model_architecture.png')
    print("NetworkX visualization saved as 'model_architecture.png'")

# 4. Snake Layout Visualization
def visualize_with_snake_layout():
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes for each layer
    layer_names = [layer.name for layer in model.layers]
    for i, name in enumerate(layer_names):
        G.add_node(name, layer_type=type(model.layers[i]).__name__)
    
    # Add edges between layers
    for i in range(len(layer_names)-1):
        G.add_edge(layer_names[i], layer_names[i+1])
    
    # Create snake layout positions
    pos = {}
    num_layers = len(layer_names)
    for i, name in enumerate(layer_names):
        # Alternate between top and bottom positions
        y = 1 if i % 2 == 0 else -1
        x = i / (num_layers - 1)  # Normalize x position between 0 and 1
        pos[name] = (x, y)
    
    # Draw the graph
    plt.figure(figsize=(15, 6))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue')
    
    # Draw edges with curved paths
    nx.draw_networkx_edges(G, pos, arrows=True, connectionstyle='arc3,rad=0.2')
    
    # Draw labels
    labels = {node: f"{node}\n({G.nodes[node]['layer_type']})" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels)
    
    plt.title("Neural Network Architecture (Snake Layout)")
    plt.axis('off')
    plt.savefig('model_architecture_snake.png')
    print("Snake layout visualization saved as 'model_architecture_snake.png'")

if __name__ == "__main__":
    print("Generating model visualizations...")
    
    # Create visualizations
    visualize_with_netron()
    visualize_with_tensorboard()
    visualize_with_networkx()
    visualize_with_snake_layout()
    
    print("\nVisualizations complete!")
    print("1. Netron: View at http://localhost:8080")
    print("2. TensorBoard: Run 'tensorboard --logdir logs/fit'")
    print("3. NetworkX: View 'model_architecture.png'")
    print("4. Snake Layout: View 'model_architecture_snake.png'") 