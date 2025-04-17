import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True, in_channels=3):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, in_channels=3):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# Configuration for different VGG variants
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class vgg:
    def vgg11(pretrained=False, batch_norm=False, **kwargs):
        model = VGG(make_layers(cfg['A'], batch_norm=batch_norm, in_channels=kwargs.get('in_channels', 3)), **kwargs)
        return model

    def vgg13(pretrained=False, batch_norm=False, **kwargs):
        model = VGG(make_layers(cfg['B'], batch_norm=batch_norm, in_channels=kwargs.get('in_channels', 3)), **kwargs)
        return model

    def vgg16(pretrained=False, batch_norm=False, **kwargs):
        model = VGG(make_layers(cfg['D'], batch_norm=batch_norm, in_channels=kwargs.get('in_channels', 3)), **kwargs)
        return model

    def vgg19(pretrained=False, batch_norm=False, **kwargs):
        model = VGG(make_layers(cfg['E'], batch_norm=batch_norm, in_channels=kwargs.get('in_channels', 3)), **kwargs)
        return model

    def vgg11_bn(pretrained=False, **kwargs):
        model = VGG(make_layers(cfg['A'], batch_norm=True, in_channels=kwargs.get('in_channels', 3)), **kwargs)
        return model

    def vgg13_bn(pretrained=False, **kwargs):
        model = VGG(make_layers(cfg['B'], batch_norm=True, in_channels=kwargs.get('in_channels', 3)), **kwargs)
        return model

    def vgg16_bn(pretrained=False, **kwargs):
        model = VGG(make_layers(cfg['D'], batch_norm=True, in_channels=kwargs.get('in_channels', 3)), **kwargs)
        return model

    def vgg19_bn(pretrained=False, **kwargs):
        model = VGG(make_layers(cfg['E'], batch_norm=True, in_channels=kwargs.get('in_channels', 3)), **kwargs)
        return model