import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbones.resnet import resnet50_3d, resnet151_3d

class UNetResNet3D(nn.Module):
    def __init__(self, resnet_type="resnet50_3d", in_channels=4, out_channels=4, dropout_rate=0.3):
        super(UNetResNet3D, self).__init__()

        # Load ResNet as Encoder
        if resnet_type == "resnet50_3d":
            self.encoder = resnet50_3d(in_channels=in_channels)
        elif resnet_type == "resnet151_3d":
            self.encoder = resnet151_3d(in_channels=in_channels)
        else:
            raise ValueError("Unsupported ResNet type. Choose 'resnet50_3d' or 'resnet151_3d'.")

        # Remove ResNet fully connected layers
        del self.encoder.avgpool
        del self.encoder.fc

        # Apply dropout for regularization
        self.dropout = nn.Dropout3d(p=dropout_rate)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm3d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_rate),
            nn.Conv3d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm3d(1024),
            nn.ReLU(inplace=True),
        )

        # Decoder (Upsampling + Convolution)
        self.upconv1 = nn.ConvTranspose3d(1024, 1024, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(1024 + 1024, 512)  # enc4 (1024) + bottleneck_upsampled (1024)

        self.upconv2 = nn.ConvTranspose3d(512, 512, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(512 + 512, 256)  # enc3 (512) + dec1_upsampled (512)

        self.upconv3 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(256 + 256, 128)  # enc2 (256) + dec2_upsampled (256)

        self.upconv4 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(128 + 64, 64)  # enc1 (64) + dec3_upsampled (128)

        self.upconv5 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        self.final = nn.Conv3d(64, out_channels, kernel_size=1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=0.2),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        enc0 = self.encoder.conv1(x)
        enc0 = self.encoder.bn1(enc0)
        enc0 = self.encoder.relu(enc0)

        enc1 = self.encoder.maxpool(enc0)  # 64 channels

        enc2 = self.encoder.layer1(enc1)  # 256 channels
        enc3 = self.encoder.layer2(enc2)  # 512 channels
        enc4 = self.encoder.layer3(enc3)  # 1024 channels
        enc5 = self.encoder.layer4(enc4)  # 2048 channels

        # Apply dropout for regularization
        enc5 = self.dropout(enc5)

        # Bottleneck
        bottleneck = self.bottleneck(enc5)  # 1024 channels

        dec1 = self.upconv1(bottleneck)
        # Resize enc4 to match dec1 spatial dimensions
        enc4_resized = F.interpolate(enc4, size=(dec1.size(2), dec1.size(3), dec1.size(4)),
                                    mode='trilinear', align_corners=False)
        dec1 = torch.cat((dec1, enc4_resized), dim=1)
        dec1 = self.dec1(dec1)

        dec2 = self.upconv2(dec1)
        # Resize enc3 to match dec2 spatial dimensions
        enc3_resized = F.interpolate(enc3, size=(dec2.size(2), dec2.size(3), dec2.size(4)),
                                    mode='trilinear', align_corners=False)
        dec2 = torch.cat((dec2, enc3_resized), dim=1)
        dec2 = self.dec2(dec2)

        dec3 = self.upconv3(dec2)
        # Resize enc2 to match dec3 spatial dimensions
        enc2_resized = F.interpolate(enc2, size=(dec3.size(2), dec3.size(3), dec3.size(4)),
                                    mode='trilinear', align_corners=False)
        dec3 = torch.cat((dec3, enc2_resized), dim=1)
        dec3 = self.dec3(dec3)

        dec4 = self.upconv4(dec3)
        # Resize enc1 to match dec4 spatial dimensions
        enc1_resized = F.interpolate(enc1, size=(dec4.size(2), dec4.size(3), dec4.size(4)),
                                    mode='trilinear', align_corners=False)
        dec4 = torch.cat((dec4, enc1_resized), dim=1)
        dec4 = self.dec4(dec4)

        dec5 = self.upconv5(dec4)
        out = self.final(dec5)

        return out


# Add tumor classification model
class TumorClassifier(nn.Module):
    def __init__(self, input_channels=4, pretrained_segmenter=None):
        super(TumorClassifier, self).__init__()

        # Use segmentation model's encoder as feature extractor
        if pretrained_segmenter is not None:
            self.encoder = pretrained_segmenter.encoder
            # Freeze encoder weights if using pretrained
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            self.encoder = resnet50_3d(in_channels=input_channels)

        # Remove unnecessary parts
        del self.encoder.fc

        # Global average pooling and classification head
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 64),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(64, 2)  # Binary classification: benign or malignant
        )

    def forward(self, x):
        # Extract features using the encoder
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)

        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # Classification
        x = self.classifier(x)

        return x