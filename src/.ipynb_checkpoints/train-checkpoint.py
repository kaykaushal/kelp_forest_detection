from tqdm import tqdm 
import torch
from src.model import calculate_dice_coefficient, calculate_iou

class Trainer:
    def __init__(self, model, train_dataloader, valid_dataloader, criterion, optimizer, device):
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, epochs):
        for epoch in range(epochs):
            self.model.train()
            loop = tqdm(self.train_dataloader, desc=f'Epoch {epoch+1}/{epochs}')
            total_loss = 0.0
            total_iou = 0.0
            total_dice = 0.0
            for batch_idx, (data, targets) in enumerate(loop):
                data, targets = data.to(self.device), targets.to(self.device)
                # foward
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                # backward
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}")
            # Validation
            self.model.eval()
            with torch.no_grad():
                total_iou = 0.0
                total_dice = 0.0
                total_loss = 0.0
                for data, targets in self.valid_dataloader:
                    data, targets = data.to(self.device), targets.to(self.device)
                    outputs = self.model(data)
                    #Loss matrix
                    val_loss = self.criterion(outputs, targets)
                    iou = calculate_iou(torch.sigmoid(outputs) > 0.5, targets > 0.5)
                    dice_coefficient = calculate_dice_coefficient(torch.sigmoid(outputs) > 0.5, targets > 0.5)
                    #Total Loss & cofficient 
                    total_iou += iou.item()
                    total_dice += dice_coefficient.item()
                    total_loss += val_loss.item()
                avg_loss = total_loss / len(self.valid_dataloader)
                avg_iou = total_iou / len(self.valid_dataloader)
                avg_dice = total_dice / len(self.valid_dataloader)
                #self.valid_losses.append(avg_loss)  # Store the validation loss
                print(f"Valid IoU: {avg_iou:.4f}, Valid Dice: {avg_dice:.4f}, Valid Loss: {avg_loss:.4f}")