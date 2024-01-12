from tqdm.auto import tqdm 
import torch
from src.model import  calculate_iou, calculate_dice_coefficient
from torch.utils.tensorboard import SummaryWriter
import os 


home_dir = "/home/jovyan/open_pluto/kelp_forest_detection"
os.chdir(home_dir)

def create_writer(experiment_name: str, 
                  model_name: str, 
                  extra: str=None) -> torch.utils.tensorboard.writer.SummaryWriter():
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """
    from datetime import datetime
    import os

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

    if extra:
        # Create log directory path
        log_dir = os.path.join("tensorboard/runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("tensorboard/runs", timestamp, experiment_name, model_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)


class Trainer:
    def __init__(self, model, train_dataloader, valid_dataloader, writer,
                 criterion, optimizer, device,tb_path, checkpoint_path, trial):
        """
        Training Process
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.writer = torch.utils.tensorboard.writer.SummaryWriter
        self.tb_path = tb_path
        self.best_valid_dice = 0.0
        self.trial = trial
    
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.criterion,
        }
        checkpoint_filename = f'models/{self.checkpoint_path}/trial_{self.trial}_checkpoint_{epoch+1}.pt'
        torch.save(checkpoint, checkpoint_filename)
        print(f"Checkpoint saved at epoch {epoch+1}")
        
        if is_best:
            best_model_filename = f'models/trial_{self.trial}_best_model.pt'
            torch.save(checkpoint, best_model_filename)
            print(f"Best model saved at epoch {epoch+1}")

    def train(self, epochs):
        writer = SummaryWriter(self.tb_path)
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            total_iou = 0.0
            total_dice = 0.0
            results = {"train_loss": [],
                      "train_dice": [],
                      "valid_loss": [],
                      "valid_dice": [],   
            }
            
            loop = tqdm(self.train_dataloader, desc=f'Epoch {epoch+1}/{epochs}')
            for batch_idx, (data, targets) in enumerate(loop):
                data, targets = data.to(self.device), targets.to(self.device)
                # foward
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                dice_coefficient = calculate_dice_coefficient(torch.sigmoid(outputs) > 0.5, 
                                                                  targets > 0.5)
                # backward
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                total_dice += dice_coefficient.item()
            avg_loss = total_loss / len(self.train_dataloader)
            avg_tdice = total_dice / len(self.train_dataloader)
            print(f"Epoch {epoch+1}/{epochs},Train Loss: {avg_loss:.4f} | Dice: {avg_tdice:.4f}")
            #writer.add_scalar('Train Loss', avg_loss, epoch)
            # Save checkpoint after each epoch
            #self.save_checkpoint(epoch) #is_best=True
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                total_iou = 0.0
                total_dice = 0.0
                total_vloss = 0.0
                for data, targets in self.valid_dataloader:
                    data, targets = data.to(self.device), targets.to(self.device)
                    outputs = self.model(data)
                    #Loss matrix
                    val_loss = self.criterion(outputs, targets)
                    iou = calculate_iou(torch.sigmoid(outputs) > 0.5, targets > 0.5)
                    dice_coefficient = calculate_dice_coefficient(torch.sigmoid(outputs) > 0.5, 
                                                                  targets > 0.5)
                    #Total Loss & cofficient 
                    total_iou += iou.item()
                    total_dice += dice_coefficient.item()
                    total_vloss += val_loss.item()
                avg_vloss = total_vloss / len(self.valid_dataloader)
                avg_iou = total_iou / len(self.valid_dataloader)
                avg_vdice = total_dice / len(self.valid_dataloader)
                
                results["train_loss"].append(avg_loss)
                results["train_dice"].append(avg_tdice)
                results["valid_loss"].append(avg_vloss)
                results["valid_dice"].append(avg_vdice)
                # Log the running loss averaged per batch
                if writer:
                    writer.add_scalars(main_tag='Loss',
                                    tag_scalar_dict = { 'train' : avg_loss, 'valid' : avg_vloss },
                                    global_step = epoch)
                    writer.add_scalars(main_tag = 'main_tagDice',
                                    tag_scalar_dict = { 'train' : avg_tdice, 'valid' : avg_vdice },
                                    global_step = epoch)
                    #writer.add_scalar('Valid Loss', val_avg_loss, epoch)
                    #writer.add_scalar('Valid Dice', avg_dice, epoch)
                    #self.valid_losses.append(avg_loss)  # Store the validation loss
                    writer.flush()
                    writer.close()
                else:
                    pass
                print(f"Valid Loss: {avg_vloss:.4f} | IoU: {avg_iou:.4f} | Dice: {avg_vdice:.4f}")
                # Check if the current model has the best validation loss
                if avg_vdice > self.best_valid_dice:
                    self.best_valid_dice = avg_vdice
                    self.save_checkpoint(epoch, is_best=True)
        print('Finished Training')
        return results
    
if __name__ == "__main__":
    example_tb_writer = create_writer("test_optuna", "unetplus", "5_epoch")
    print(example_tb_writer)