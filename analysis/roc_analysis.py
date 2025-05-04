import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pose_classifier import PoseClassifier, PoseSequenceDataset

def create_data_loader(positive_dir, negative_dir, batch_size=32, shuffle=False):
    """
    Create DataLoader for the specified dataset.
    
    Args:
        positive_dir: Directory containing positive samples
        negative_dir: Directory containing negative samples
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
    Returns:
        data_loader
    """
    # Create dataset
    dataset = PoseSequenceDataset(
        positive_dir=positive_dir,
        negative_dir=negative_dir
    )
    
    # Create DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4
    )
    
    return data_loader

def generate_roc_curve(model, data_loader, device, dataset_name, save_path='analysis_results/roc_curves.png'):
    """
    Generate and plot ROC curve for the model.
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader for the data
        device: Device to run model on
        dataset_name: Name of the dataset (for labeling)
        save_path: Path to save the ROC curve plot
    """
    # Create directory for results if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc=f"Generating ROC curve for {dataset_name}"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            positive_probs = probabilities[:, 1].cpu().numpy()
            
            all_predictions.extend(positive_probs)
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_predictions)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc

def plot_roc_curves(roc_data, save_path='analysis_results/roc_curves.png'):
    """
    Plot multiple ROC curves on the same figure with scientific formatting.
    
    Args:
        roc_data: Dictionary containing ROC curve data for each dataset
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Create a scientific color scheme
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(roc_data) + 1))
    
    # Plot random baseline
    plt.plot([0, 1], [0, 1], color=colors[0], lw=2, linestyle='--', label='Random')
    
    # Plot ROC curves for each dataset
    for i, (dataset_name, (fpr, tpr, roc_auc)) in enumerate(roc_data.items()):
        plt.plot(fpr, tpr, color=colors[i + 1], lw=2, 
                label=f'{dataset_name} (AUC = {roc_auc:.3f})')
    
    # Set axis limits and labels
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    
    # Add title with padding
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14, pad=20)
    
    # Add legend with scientific formatting
    plt.legend(loc="lower right", fontsize=10, framealpha=0.9)
    
    # Add grid with subtle styling
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save with high DPI
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save the ROC curve data
    np.savez(save_path.replace('.png', '_data.npz'),
             **{f'{name}_fpr': fpr for name, (fpr, _, _) in roc_data.items()},
             **{f'{name}_tpr': tpr for name, (_, tpr, _) in roc_data.items()},
             **{f'{name}_auc': auc for name, (_, _, auc) in roc_data.items()})

def main():
    # Create model
    model = PoseClassifier()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pre-trained model
    print("Loading pre-trained model...")
    model.load_state_dict(torch.load('models/best_model.pth'))
    model = model.to(device)
    
    # Create data loaders
    train_loader = create_data_loader('rat_dance_csv/train', 'neg_control_csv/train')
    test_loader = create_data_loader('rat_dance_csv/test', 'neg_control_csv/test')
    
    # Generate ROC curves
    roc_data = {}
    
    # Generate ROC curve for training data
    fpr, tpr, roc_auc = generate_roc_curve(model, train_loader, device, 'Training')
    roc_data['Training'] = (fpr, tpr, roc_auc)
    print(f"Training ROC AUC Score: {roc_auc:.4f}")
    
    # Generate ROC curve for test data
    fpr, tpr, roc_auc = generate_roc_curve(model, test_loader, device, 'Test')
    roc_data['Test'] = (fpr, tpr, roc_auc)
    print(f"Test ROC AUC Score: {roc_auc:.4f}")
    
    # Plot and save ROC curves
    plot_roc_curves(roc_data)

if __name__ == '__main__':
    main() 