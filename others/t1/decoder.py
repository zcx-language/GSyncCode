import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return self.relu(x + residual)
    
class EnhanceNet(nn.Module):
    def __init__(self):
        super(EnhanceNet, self).__init__()
        # Define the layers of the network
        # ...

        self.layer_1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU()
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU()
        )
        self.layer_3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU()
        )

        self.layer_4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU()
        )

        self.mid_layer = nn.Sequential(
            nn.MaxPool2d((2, 2)),
            nn.Upsample(scale_factor=2)
            )
        self.down_layer = nn.MaxPool2d((2, 2))
        self.up_layer = nn.Upsample(scale_factor=2)

    
        self.up_layer_4 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU()
        )
        self.up_layer_3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU()
        )
        self.up_layer_2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU()
        )
        self.up_layer_1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 1, 1)
        )
    def forward(self, x):
        # Define the forward pass of the network
        # ...

        x1 = self.layer_1(x)
        x2 = self.layer_2(self.down_layer(x1))
        x3 = self.layer_3(self.down_layer(x2))
        x4 = self.layer_4(self.down_layer(x3))
        mid_x = self.mid_layer(x4)
        y4 = self.up_layer_4(torch.cat((x4, mid_x), dim=1))
        y3 = self.up_layer_3(torch.cat((x3, self.up_layer(y4)), dim=1))
        y2 = self.up_layer_2(torch.cat((x2, self.up_layer(y3)), dim=1))
        y1 = self.up_layer_1(torch.cat((x1, self.up_layer(y2)), dim=1))
        return y1


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.regression_1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.MaxPool2d((2, 2)),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.MaxPool2d((2, 2)),
        )
        self.enhance = EnhanceNet()
        self.att = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            ResidualBlock(64),
            nn.MaxPool2d((2, 2)),
            ResidualBlock(64),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU()
        )

        self.regression_2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.MaxPool2d((2, 2)),
            ResidualBlock(64),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 1, 1, 1),
            nn.ReLU(),
        )
        pass

    def forward(self, x):
        enhance_x = self.enhance(x)
        att_feat = self.att(x)
        feat = self.regression_1(enhance_x)
        feat = feat * att_feat
        msg = self.regression_2(feat)
        return msg
        pass
    pass
if __name__ == "__main__":
    model = Decoder()
    x = torch.randn(1, 3, 256, 256)
    print(model(x).size())