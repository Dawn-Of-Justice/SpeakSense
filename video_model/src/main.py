import torch
from src.data.video_dataset import GRIDDataset
from src.models.video_cnn import VideoCNN
from src.training.train import train_video_model
from torch.utils.data import DataLoader

def main():
    # Configuration
    data_root = "video_model\src\data\dataset\GRID\video"
    train_subjects = list(range(1, 26)).remove(21)  # subjects 1-25 for training
    val_subjects = list(range(26, 30))   # subjects 26-29 for validation
    
    # Create datasets
    train_dataset = GRIDDataset(data_root, train_subjects)
    val_dataset = GRIDDataset(data_root, val_subjects)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = VideoCNN(num_classes=2)
    
    # Train model
    trained_model = train_video_model(model, train_loader, val_loader)
    
    # Save model
    torch.save(trained_model.state_dict(), 'grid_video_model.pth')

if __name__ == "__main__":
    main()