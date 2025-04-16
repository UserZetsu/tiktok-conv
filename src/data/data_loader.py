import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os

class PoseSequenceDataset(Dataset):
    def __init__(self, positive_dir, negative_dir, sequence_length=20):
        """
        Initialize the dataset.
        
        Args:
            positive_dir (str): Directory containing positive (rat dance) samples
            negative_dir (str): Directory containing negative control samples
            sequence_length (int): Number of frames to include in each sequence
        """
        self.sequence_length = sequence_length
        self.samples = []
        self.labels = []
        
        # Load positive samples
        for file in os.listdir(positive_dir):
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(positive_dir, file))
                self.samples.append(df)
                self.labels.append(1)  # 1 for rat dance
        
        # Load negative samples
        for file in os.listdir(negative_dir):
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(negative_dir, file))
                self.samples.append(df)
                self.labels.append(0)  # 0 for negative control
        
        # Convert labels to tensor
        self.labels = torch.tensor(self.labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Get the sequence data
        df = self.samples[idx]
        
        # Extract pose features (excluding frame_id, person_id, and confidence columns)
        pose_columns = [col for col in df.columns if col not in ['frame_id', 'person_id', 'confidence']]
        pose_data = df[pose_columns].values
        
        # Pad or truncate to sequence_length
        if len(pose_data) > self.sequence_length:
            pose_data = pose_data[:self.sequence_length]
        else:
            padding = np.zeros((self.sequence_length - len(pose_data), pose_data.shape[1]))
            pose_data = np.vstack([pose_data, padding])
        
        # Convert to tensor
        pose_data = torch.tensor(pose_data, dtype=torch.float32)
        
        return pose_data, self.labels[idx]

def get_data_loaders(positive_dir, negative_dir, batch_size=32, sequence_length=20, val_split=0.2):
    """
    Create training and validation data loaders.
    
    Args:
        positive_dir (str): Directory containing positive samples
        negative_dir (str): Directory containing negative samples
        batch_size (int): Batch size for data loaders
        sequence_length (int): Number of frames in each sequence
        val_split (float): Fraction of data to use for validation
        
    Returns:
        train_loader, val_loader: PyTorch DataLoader objects
    """
    # Create dataset
    dataset = PoseSequenceDataset(positive_dir, negative_dir, sequence_length)
    
    # Calculate split sizes
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    # Split dataset
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader 