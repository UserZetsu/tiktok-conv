import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

class PoseClassifier(nn.Module):
    def __init__(self, input_size=20, hidden_size=128, num_layers=2, num_classes=2):
        super(PoseClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer with hidden size 128 (matches checkpoint)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        # Fully connected layers with sizes matching checkpoint
        self.fc1 = nn.Linear(hidden_size * 2, 128)  # 256 -> 128
        self.fc2 = nn.Linear(128, num_classes)  # 128 -> 2
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

def extract_features(data):
    """Extract pose landmarks from the data."""
    features = []
    for df in data:
        # Extract pose coordinates (x and y for each landmark)
        pose_features = []
        for i in range(10):  # 10 pose landmarks
            x_col = f'x_pose_{i}'
            y_col = f'y_pose_{i}'
            pose_features.extend([df[x_col].values, df[y_col].values])
        
        # Stack features for each frame
        frame_features = np.stack(pose_features, axis=1)
        features.append(frame_features)
    
    return np.array(features)

def calculate_importance_scores(model, data, labels):
    """Calculate importance scores for each landmark using multiple metrics."""
    model.eval()
    device = next(model.parameters()).device
    
    # Convert data to tensor
    data_tensor = torch.FloatTensor(data).to(device)
    
    # Get model predictions
    with torch.no_grad():
        predictions = model(data_tensor)
        predictions = torch.softmax(predictions, dim=1)
        pred_scores = predictions[:, 1].cpu().numpy()  # Get positive class probabilities
    
    # Calculate importance scores
    importance_scores = np.zeros((data.shape[2] // 2, 3))  # 3 metrics per landmark
    
    for landmark_idx in range(data.shape[2] // 2):
        # Movement magnitude
        x_coords = data[:, :, landmark_idx * 2]
        y_coords = data[:, :, landmark_idx * 2 + 1]
        movement = np.sqrt(np.diff(x_coords, axis=1)**2 + np.diff(y_coords, axis=1)**2)
        importance_scores[landmark_idx, 0] = np.mean(movement)
        
        # Correlation with predictions
        # Use mean position of landmark across time
        x_mean = np.mean(x_coords, axis=1)
        y_mean = np.mean(y_coords, axis=1)
        x_corr = np.abs(np.corrcoef(x_mean, pred_scores)[0, 1])
        y_corr = np.abs(np.corrcoef(y_mean, pred_scores)[0, 1])
        importance_scores[landmark_idx, 1] = (x_corr + y_corr) / 2
        
        # Prediction change when landmark is perturbed
        perturbed_data = data_tensor.clone()
        noise = torch.randn_like(perturbed_data[:, :, landmark_idx * 2:landmark_idx * 2 + 2]) * 0.1
        perturbed_data[:, :, landmark_idx * 2:landmark_idx * 2 + 2] += noise
        
        with torch.no_grad():
            perturbed_predictions = model(perturbed_data)
            perturbed_predictions = torch.softmax(perturbed_predictions, dim=1)
        
        importance_scores[landmark_idx, 2] = torch.mean(torch.abs(predictions - perturbed_predictions)).item()
    
    return importance_scores

def plot_importance_scores(importance_scores, save_path='landmark_importance.png'):
    """Plot and save importance scores."""
    plt.figure(figsize=(15, 5))
    
    # Plot each metric
    metrics = ['Movement Magnitude', 'Correlation with Predictions', 'Prediction Change']
    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i + 1)
        sns.barplot(x=np.arange(len(importance_scores)), y=importance_scores[:, i])
        plt.title(metric)
        plt.xlabel('Landmark Index')
        plt.ylabel('Importance Score')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    # Save importance scores to CSV
    df = pd.DataFrame(importance_scores, columns=metrics)
    df.to_csv('landmark_importance.csv', index=False)

def main():
    # Load the trained model
    model = PoseClassifier()
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    # Load and preprocess data
    pos_dir = 'rat_dance_csv/train'
    neg_dir = 'neg_control_csv/train'
    data, labels = load_and_preprocess_data(pos_dir, neg_dir)
    
    # Extract features
    features = extract_features(data)
    
    # Calculate importance scores
    importance_scores = calculate_importance_scores(model, features, labels)
    
    # Plot and save results
    plot_importance_scores(importance_scores)

if __name__ == '__main__':
    main() 