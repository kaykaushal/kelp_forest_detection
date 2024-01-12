import os 
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torchvision 
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

import optuna
from src.dataset import KelpDataset
from src.model import BaseUnet, UNetPlus
from src.train import Trainer

HOME_DIR = '/home/jovyan/open_pluto/kelp_forest_detection'
os.chdir(HOME_DIR)
BATCH_SIZE = 4
WORKER = 2
CHANNEL_IN = 7
CHANNEL_OUT = 1
IMAGE_DIR = "data/training/train_satellite"
MASK_DIR = "data/training/train_kelp"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Get metdatafile for training and test split 
meta_df = pd.read_csv('metadata_kelp.csv')
df = meta_df[meta_df['in_train'] == True].head(500)
train_df, valid_df = train_test_split(df, test_size = .2, random_state=42)
train_files = train_df['filename'].tolist()
valid_files = valid_df['filename'].tolist()
## Set tensor dataset and loader
torch.manual_seed(12)
transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Resize((256, 256)),
])
# Custome datasets
# Create a custom training custom dataset for training dataset
train_dataset = KelpDataset(image_dir=IMAGE_DIR, mask_dir=MASK_DIR,
                            transform=transform, filename_list=train_files)
# Create a custom training custom dataset for validation set
valid_dataset = KelpDataset(image_dir=IMAGE_DIR, mask_dir=MASK_DIR,
                            transform=transform, filename_list=valid_files)
# Load dataloader for train and validation set
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=4, pin_memory=True,  prefetch_factor=2)
# Validation loader
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                              num_workers=4, pin_memory=True,  prefetch_factor=WORKER)

# Define the objective function for Optuna
def objective(trial):
    # Define hyperparameters to be optimized
    in_channels = 7  # Change if your input channels are different
    out_channels = 1
    features = [trial.suggest_int(f'features_{i}', 32, 512) for i in range(4)]
    dropout_prob = trial.suggest_float('dropout_prob', 0.0, 0.5)
    # Define model
    model = UNetPlus(
        in_channels=in_channels,
        out_channels=out_channels,
        features=features,
        dropout_prob=dropout_prob
    )
    
    # Set training hyperparameter
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    # Choose optimizer and its hyperparameters
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'Adamax'])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adamax(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Choose loss function
    loss_name = trial.suggest_categorical('loss', ['MSELoss', 'HuberLoss', 'BCELoss'])
    if loss_name == 'MSELoss':
        criterion = torch.nn.MSELoss()
    elif loss_name == 'HuberLoss':
        criterion = torch.nn.HuberLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate Trainer with Optuna trial
    trainer = Trainer(
        model=model.to(device),
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        tb_path='tensorboard/runs/',
        checkpoint_path='tune/',
        trial=trial
    )
    
    # Train the model
    trainer.train(epochs=10)  # Adjust the number of epochs as needed

    # Return a scalar value indicating the performance
    return trainer.best_valid_loss

if __name__ == "__main__":
    print(len(train_dataloader))
    # Create an optimization study and perform optimization
    study = optuna.create_study(direction='minimize')  # Optimize for minimum validation loss
    study.optimize(objective, n_trials=10)