import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms.functional as TF
from torchvision.models.segmentation import deeplabv3_resnet101

class DoubleConv(nn.Module):
    """
    Double convolution block with batch normalization and ReLU activation.
    
    Parameters:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels, dropout_prob=0.0):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob),
        )

    def forward(self, x):
        return self.conv(x)

class BaseUnet(nn.Module):
    """
    Base UNet model for semantic segmentation of kelp.
    
    Parameters:
        - in_channels (int): Number of input channels (default is 3 for Landsat bands).
        - out_channels (int): Number of output channels (default is 1 for binary segmentation).
        - features (list): List of features for each level of the UNet (default is [64, 128, 256, 512]).
    """
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256], dropout_prob=0.0):
        super(BaseUnet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature, dropout_prob=dropout_prob))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the UNet model.
        
        Parameters:
            - x (torch.Tensor): Input tensor.
            
        Returns:
            - torch.Tensor: Output tensor.
        """
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)  # Fix dimension along channels
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

# Custom DiceLoss    
class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        # Flatten the prediction and target tensors
        prediction_flat = prediction.view(-1)
        target_flat = target.view(-1)

        intersection = (prediction_flat * target_flat).sum()
        union = prediction_flat.sum() + target_flat.sum()

        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)

        # The Dice Loss is 1 - Dice Score
        dice_loss = 1 - dice_score

        return dice_loss

# Custom loss functions (IoU and Dice coefficient)
def calculate_iou(outputs, targets):
    """
    Calculate the Intersection over Union (IoU) for binary segmentation.
    
    Parameters:
        - outputs (torch.Tensor): Model predictions.
        - targets (torch.Tensor): Ground truth labels.
        
    Returns:
        - torch.Tensor: Mean IoU.
    """
    intersection = torch.logical_and(outputs, targets).sum()
    union = torch.logical_or(outputs, targets).sum()
    iou = (intersection + 1e-10) / (union + 1e-10)
    return iou.mean()

def calculate_dice_coefficient(outputs, targets):
    """
    Calculate the Dice coefficient for binary segmentation.
    
    Parameters:
        - outputs (torch.Tensor): Model predictions.
        - targets (torch.Tensor): Ground truth labels.
        
    Returns:
        - torch.Tensor: Mean Dice coefficient.
    """
    intersection = torch.logical_and(outputs, targets).sum()
    dice_coefficient = (2 * intersection + 1e-10) / (outputs.sum() + targets.sum() + 1e-10)
    return dice_coefficient.mean()




###################################### Transfer Learnig based model ###################
## Transfer Learning with U-Net
class TransferUnet(nn.Module):
    def __init__(self, unet_model, in_channels=3, out_channels=1, features=[32, 64, 128, 256], dropout_prob=0.0):
        super(TransferUnet, self).__init__()

        # Load your pre-trained U-Net model
        self.pretrained_unet = unet_model

        # Adjusting the weights for compatibility with the desired input channels
        self.adjust_weights(in_channels)

        # Replace the last layer for binary segmentation
        self.pretrained_unet.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        # Custom decoder
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature, dropout_prob=dropout_prob))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def adjust_weights(self, in_channels):
        pretrained_weights = self.pretrained_unet.downs[0].conv[0].weight.data
        new_weights = torch.zeros(pretrained_weights.size(0), in_channels, pretrained_weights.size(2), pretrained_weights.size(3))
        new_weights[:, :in_channels, :, :] = pretrained_weights[:, :in_channels, :, :]
        self.pretrained_unet.downs[0].conv[0].weight.data = new_weights

    def forward(self, x):
        # Forward pass through the pre-trained U-Net
        pretrained_features = self.pretrained_unet(x)

        # Custom decoder
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        # Final convolution layer
        output = self.final_conv(x)

        return output

#####Unet++ #########
class UNetPlus(nn.Module):
    """
    U-Net++ model for semantic segmentation.
    
    Parameters:
        - in_channels (int): Number of input channels (default is 3 for RGB images).
        - out_channels (int): Number of output channels (default is 1 for binary segmentation).
        - features (list): List of features for each level of the U-Net++ (default is [32, 64, 128, 256]).
        - dropout_prob (float): Dropout probability (default is 0.0).
    """
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], dropout_prob=0.0):
        super(UNetPlus, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of U-Net++
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature, dropout_prob=dropout_prob))
            in_channels = feature

        # Up part of U-Net++
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature, dropout_prob=dropout_prob))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2, dropout_prob=dropout_prob)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the U-Net++ model.
        
        Parameters:
            - x (torch.Tensor): Input tensor.
            
        Returns:
            - torch.Tensor: Output tensor.
        """
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

class SegNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], BN_momentum=0.5):
        super(SegNet, self).__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        in_channels_encoder = in_channels
        for feature in features:
            self.downs.append(DoubleConv(in_channels_encoder, feature, dropout_prob=0.0))
            in_channels_encoder = feature

        in_channels_decoder = features[-1] * 2
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    in_channels_decoder, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(in_channels_decoder, feature, dropout_prob=0.0))
            in_channels_decoder = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, dropout_prob=0.0)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)

def main():
    # Example code for running and testing the models
    image_data = torch.randn((4, 3, 350, 350))  # Example image data
    model_0 = BaseUnet(in_channels=7, out_channels=1)
    model_1 = UNetPlus(in_channels=3, out_channels=1)
    pred = model_1(image_data)
    print(pred.shape)
    
# Check if the script is being run as the main program
if __name__ == "__main__":
    # Call the main function
    main()