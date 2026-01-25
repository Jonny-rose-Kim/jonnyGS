"""
Simple U-Net for noise detection
"""

import torch
import torch.nn as nn


class SimpleUNet(nn.Module):
    """Simple U-Net architecture for binary segmentation"""

    def __init__(self, in_channels=3, out_channels=1):
        super(SimpleUNet, self).__init__()

        # Encoder
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._block(128, 64)

        # Output
        self.out = nn.Conv2d(64, out_channels, 1)

        # Pooling
        self.pool = nn.MaxPool2d(2)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Decoder
        d3 = self.up3(e4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Output
        out = self.out(d1)
        return torch.sigmoid(out)


class DeepUNet(nn.Module):
    """Deeper U-Net architecture for better feature extraction"""

    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512, 1024]):
        super(DeepUNet, self).__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(0.1)

        # Encoder path
        for i, feature in enumerate(features):
            in_ch = in_channels if i == 0 else features[i - 1]
            self.encoders.append(self._block(in_ch, feature))

        # Decoder path
        for i in range(len(features) - 1, 0, -1):
            self.ups.append(nn.ConvTranspose2d(features[i], features[i - 1], 2, stride=2))
            self.decoders.append(self._block(features[i], features[i - 1]))

        # Output
        self.out = nn.Conv2d(features[0], out_channels, 1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        skip_connections = []

        # Encoder
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            if i < len(self.encoders) - 1:
                skip_connections.append(x)
                x = self.pool(x)
                x = self.dropout(x)

        skip_connections = skip_connections[::-1]

        # Decoder
        for i, (up, decoder) in enumerate(zip(self.ups, self.decoders)):
            x = up(x)
            skip = skip_connections[i]
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        return torch.sigmoid(self.out(x))
