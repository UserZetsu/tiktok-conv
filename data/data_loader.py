import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os

class PoseSequenceDataset(Dataset):
    def __init__(self, positive_dir, negative_dir, sequence_length=20):
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

def create_data_loaders(positive_dir, negative_dir, batch_size=32, sequence_length=20):
    """Create training and validation data loaders.
    
    Args:
        positive_dir: Directory containing positive samples
        negative_dir: Directory containing negative samples
        batch_size: Batch size for data loaders
        sequence_length: Length of pose sequences
        
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
    """
    # Create dataset
    dataset = PoseSequenceDataset(
        positive_dir=positive_dir,
        negative_dir=negative_dir,
        sequence_length=sequence_length
    )
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader 