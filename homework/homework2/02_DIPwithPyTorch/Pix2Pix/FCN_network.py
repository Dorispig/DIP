import torch.nn as nn

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        channels = [3, 8, 128, 512]
         # Encoder (Convolutional Layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True)
        )
        ### FILL: add more CONV Layers
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], kernel_size=4, stride=2, padding=1),  # Input channels: 8, Output channels: 64
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels[2], channels[3], kernel_size=4, stride=2, padding=1),  # Input channels: 64, Output channels: 256
            nn.BatchNorm2d(channels[3]),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (Deconvolutional Layers)
        ### FILL: add ConvTranspose Layers
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(channels[3], channels[2], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(channels[2], channels[1], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(channels[1], channels[0], kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(channels[0]),
            # nn.Sigmoid()
            nn.Tanh()
        )
        ### Note: since last layer outputs RGB channels, may need specific activation function

    def forward(self, x):
        # Encoder forward pass
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Decoder forward pass
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        ### FILL: encoder-decoder forward pass
        output = x
        
        return output
    