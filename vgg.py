import torch.nn as nn
from torchvision.models import vgg19


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.vgg = vgg19(pretrained=True, progress=True).features


    def get_content_features(self, x):
        content_features = self.vgg[:23](x)
        return content_features

    def get_style_features(self, x):
        style_features = [self.vgg[:4](x)] + [self.vgg[:7](x)] + [self.vgg[:12](x)] + [self.vgg[:21](x)] + [
            self.vgg[:30](x)]
        return style_features

    def forward(self, x):
        return self.vgg(x)
