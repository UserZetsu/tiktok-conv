import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def load_model(model_path):
    """Load a trained model from checkpoint."""
    model = PoseClassifier()
    checkpoint = torch.load(model_path)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def analyze_weights(model):
    """Analyze feature importance based on L2 norm of weights."""
    # Get weights from the first fully connected layer
    weights = model.fc1.weight.detach().numpy()
    
    # Calculate L2 norm for each feature
    feature_importance = np.linalg.norm(weights, axis=0)
    
    return feature_importance

def plot_feature_importance(feature_importance, feature_names):
    """Plot feature importance scores."""
    plt.figure(figsize=(12, 8))
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=True)
    
    # Create a more scientific color scheme
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(importance_df)))
    
    # Create horizontal bar plot
    bars = plt.barh(range(len(importance_df)), importance_df['Importance'], color=colors)
    
    # Format feature names to be more concise
    formatted_names = []
    for name in importance_df['Feature']:
        if name.startswith('x_pose_'):
            idx = name.split('_')[-1]
            formatted_names.append(f'X pos. of Point {idx}')
        elif name.startswith('y_pose_'):
            idx = name.split('_')[-1]
            formatted_names.append(f'Y pos. of Point {idx}')
    
    plt.yticks(range(len(importance_df)), formatted_names, fontsize=10)
    plt.title('Feature Importance', fontsize=14, pad=20)
    plt.xlabel('Weight', fontsize=12)
    plt.ylabel('Pose Features', fontsize=12)
    
    # Add grid and adjust layout
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Save with high DPI
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load the trained model
    model = load_model('best_model.pth')
    
    # Get feature names
    feature_names = [f'x_pose_{i}' for i in range(10)] + [f'y_pose_{i}' for i in range(10)]
    
    # Analyze weights
    feature_importance = analyze_weights(model)
    
    # Plot results
    plot_feature_importance(feature_importance, feature_names)
    
    # Save importance scores to CSV
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    importance_df.to_csv('feature_importance.csv', index=False)

if __name__ == '__main__':
    main() 