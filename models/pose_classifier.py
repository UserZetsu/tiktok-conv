import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class PoseSequenceDataset(Dataset):
    def __init__(self, positive_dir, negative_dir, sequence_length=357):
        self.positive_dir = positive_dir
        self.negative_dir = negative_dir
        self.sequence_length = sequence_length
        
        # Get all CSV files
        self.positive_files = [os.path.join(positive_dir, f) for f in os.listdir(positive_dir) if f.endswith('.csv')]
        self.negative_files = [os.path.join(negative_dir, f) for f in os.listdir(negative_dir) if f.endswith('.csv')]
        
        # Combine all files with their labels
        self.files = self.positive_files + self.negative_files
        self.labels = [1] * len(self.positive_files) + [0] * len(self.negative_files)
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Extract pose features (x, y coordinates for all pose points)
        pose_features = []
        for i in range(10):  # 10 pose points
            x_col = f'x_pose_{i}'
            y_col = f'y_pose_{i}'
            pose_features.extend([df[x_col].values, df[y_col].values])
        
        # Convert to numpy array and ensure correct shape
        features = np.array(pose_features).T  # Shape: (sequence_length, num_features)
        
        # Pad or truncate to sequence_length
        if features.shape[0] < self.sequence_length:
            pad_length = self.sequence_length - features.shape[0]
            features = np.pad(features, ((0, pad_length), (0, 0)), mode='constant')
        else:
            features = features[:self.sequence_length]
        
        return torch.FloatTensor(features), torch.LongTensor([label])

class PoseClassifier(nn.Module):
    def __init__(self, input_size=20, hidden_size=128, num_layers=2, num_classes=2):
        super(PoseClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
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

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels.squeeze()).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.squeeze())
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels.squeeze()).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, 'best_model.pth')
            print('Best model saved!')
        
        print('-' * 50)

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create dataset
    dataset = PoseSequenceDataset(
        positive_dir='rat_dance_csv/train',
        negative_dir='neg_control_csv/train'
    )
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create and train model
    model = PoseClassifier()
    train_model(model, train_loader, val_loader)

if __name__ == '__main__':
    main() 