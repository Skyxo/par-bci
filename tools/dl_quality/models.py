import torch
import torch.nn as nn

class EEGNetv4(nn.Module):
    """
    EEGNet v4 from Lawhern et al. 2018.
    Implemented in PyTorch.
    """
    def __init__(self, n_classes=4, n_channels=8, n_times=1000, 
                 F1=8, D=2, F2=16, kernel_length=64, dropout=0.5):
        super(EEGNetv4, self).__init__()
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernel_length = kernel_length
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_times = n_times

        # Block 1: Temporal Conv
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        
        # Block 2: Spatial Conv (Depthwise)
        self.conv2 = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.elu = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)
        
        # Block 3: Separable Conv
        self.conv3 = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), groups=F1 * D, bias=False)
        self.conv4 = nn.Conv2d(F2, F2, (1, 1), bias=False) # Pointwise
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)
        
        # Classification Layer
        # Calculate feature size
        # Time dimension reduction: 
        # Init: n_times
        # Pool1 (/4): n_times // 4
        # Pool2 (/8): (n_times // 4) // 8 = n_times // 32
        out_time = n_times // 32
        self.classifier = nn.Linear(F2 * out_time, n_classes)

    def forward(self, x):
        # Input: (Batch, Channels, Time) -> (Batch, 1, Channels, Time)
        x = x.unsqueeze(1)
        
        # Block 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.elu(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.batchnorm3(x)
        x = self.elu(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classify
        x = self.classifier(x)
        return x

    def get_spatial_weights(self):
        """
        Returns the weights of the spatial convolution layer.
        Shape: (F1 * D, 1, n_channels, 1) -> (n_filters, n_channels)
        """
        weights = self.conv2.weight.data.cpu().numpy()
        # Initial shape: (F1*D, 1, n_channels, 1)
        # Squeeze to (F1*D, n_channels)
        return weights.squeeze()
