import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

def analyze_model_performance(model, test_loader, device):
    """Analyze model performance on test data."""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    cm = confusion_matrix(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions, target_names=['Negative', 'Positive'])
    
    return cm, report

def plot_confusion_matrix(cm, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def analyze_feature_importance(model):
    """Analyze feature importance based on model weights."""
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

def analyze_sequence_importance(model, test_loader, device):
    """Analyze importance of different parts of the sequence."""
    model.eval()
    sequence_importance = np.zeros(357)  # Assuming sequence length of 357
    
    with torch.no_grad():
        for inputs, _ in tqdm(test_loader, desc="Analyzing sequences"):
            inputs = inputs.to(device)
            
            # Get model predictions
            outputs = model(inputs)
            predictions = torch.softmax(outputs, dim=1)
            
            # Calculate gradient of predictions w.r.t. input
            for i in range(inputs.size(1)):  # For each time step
                inputs.requires_grad = True
                outputs = model(inputs)
                predictions = torch.softmax(outputs, dim=1)
                
                # Calculate gradient of positive class probability
                grad = torch.autograd.grad(predictions[:, 1].sum(), inputs)[0]
                sequence_importance[i] += torch.mean(torch.abs(grad[:, i, :])).item()
    
    # Normalize importance scores
    sequence_importance /= len(test_loader)
    
    return sequence_importance

def plot_sequence_importance(sequence_importance, save_path='sequence_importance.png'):
    """Plot sequence importance scores."""
    plt.figure(figsize=(12, 6))
    plt.plot(sequence_importance)
    plt.title('Sequence Importance')
    plt.xlabel('Time Step')
    plt.ylabel('Importance Score')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Load the trained model
    model = PoseClassifier()
    model.load_state_dict(torch.load('best_model.pth'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load test data
    test_loader = create_test_loader()  # You need to implement this function
    
    # Analyze model performance
    cm, report = analyze_model_performance(model, test_loader, device)
    print("Classification Report:")
    print(report)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm)
    
    # Analyze feature importance
    feature_names = [f'x_pose_{i}' for i in range(10)] + [f'y_pose_{i}' for i in range(10)]
    feature_importance = analyze_feature_importance(model)
    plot_feature_importance(feature_importance, feature_names)
    
    # Analyze sequence importance
    sequence_importance = analyze_sequence_importance(model, test_loader, device)
    plot_sequence_importance(sequence_importance)

if __name__ == '__main__':
    main() 