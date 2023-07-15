import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, nClasses, input_height=128, input_width=128, n_filters=16, dropout=0.1, batchnorm=True):
        super(FCN, self).__init__()

        self.n_filters = n_filters

        self.block1_conv1 = nn.Conv2d(3, n_filters, kernel_size=3, padding=1)
        self.block1_conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)

        self.block2_conv1 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)
        self.block2_conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)

        self.out = nn.Conv2d(n_filters, nClasses, kernel_size=3, padding=1)

    def forward(self, x):
        # Block 1
        x = nn.ReLU()(self.block1_conv1(x))
        x = nn.ReLU()(self.block1_conv2(x))
        f1 = x

        # Block 2
        x = nn.ReLU()(self.block2_conv1(x))
        x = nn.ReLU()(self.block2_conv2(x))
        f2 = x

        out = nn.ReLU()(self.out(x))

        return out

# model = FCN(nClasses, input_height=128, input_width=128, n_filters=16, dropout=0.1, batchnorm=True)

class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, batchnorm=True, dropout=0.1):
        super(Conv2DBlock, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1))
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size, padding=1))
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(dropout))
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)

def center_crop(x, target_size):
    """ ConvTranspose2d outputs one additional dim """
    _, _, h, w = target_size
    _, _, th, tw = x.size()
    dh = (th - h) // 2
    dw = (tw - w)
    cropped = x[:, :, dh: dh + h, dw: dw + w]
    return cropped

class UNet(nn.Module):
    """
    number of input filters and channels determine the version (10, 3c or light-3c)
    """
    def __init__(self, n_classes, in_channels=10, n_filters=16, dropout=0.1, batchnorm=True):
        super(UNet, self).__init__()
        # contracting path
        self.down1 = Conv2DBlock(in_channels, n_filters, batchnorm=batchnorm, dropout=dropout)
        self.down2 = Conv2DBlock(n_filters, n_filters * 2, batchnorm=batchnorm, dropout=dropout)
        self.down3 = Conv2DBlock(n_filters * 2, n_filters * 4, batchnorm=batchnorm, dropout=dropout)
        self.down4 = Conv2DBlock(n_filters * 4, n_filters * 8, batchnorm=batchnorm, dropout=dropout)
        self.center = Conv2DBlock(n_filters * 8, n_filters * 16, batchnorm=batchnorm, dropout=dropout)
        # expansive path
        self.trans4 = nn.ConvTranspose2d(n_filters * 16, n_filters * 8, kernel_size=3, stride=2, padding=0)
        self.up4 = Conv2DBlock(n_filters * 8 + n_filters * 8, n_filters * 8, batchnorm=batchnorm, dropout=dropout)
        self.trans3 = nn.ConvTranspose2d(n_filters * 8, n_filters * 4, kernel_size=3, stride=2, padding=0)
        self.up3 = Conv2DBlock(n_filters * 4 + n_filters * 4, n_filters * 4, batchnorm=batchnorm, dropout=dropout)
        self.trans2 = nn.ConvTranspose2d(n_filters * 4, n_filters * 2, kernel_size=3, stride=2, padding=0)
        self.up2 = Conv2DBlock(n_filters * 2 + n_filters * 2, n_filters * 2, batchnorm=batchnorm, dropout=dropout)
        self.trans1 = nn.ConvTranspose2d(n_filters * 2, n_filters * 1, kernel_size=3, stride=2, padding=0)
        self.up1 = Conv2DBlock(n_filters + n_filters, n_filters, batchnorm=batchnorm, dropout=dropout)

        self.final_conv = nn.Conv2d(n_filters, n_classes, kernel_size=1)

    def forward(self, x):
        # Downsample
        d1 = self.down1(x)
        d2 = self.down2(nn.MaxPool2d(2)(d1))
        d3 = self.down3(nn.MaxPool2d(2)(d2))
        d4 = self.down4(nn.MaxPool2d(2)(d3))
        # Center
        c = self.center(nn.MaxPool2d(2)(d4))
        # Upsample
        u4 = self.up4(torch.cat([center_crop(self.trans4(c), d4.size()), d4], dim=1))
        u3 = self.up3(torch.cat([center_crop(self.trans3(u4), d3.size()), d3], dim=1))
        u2 = self.up2(torch.cat([center_crop(self.trans2(u3), d2.size()), d2], dim=1))
        u1 = self.up1(torch.cat([center_crop(self.trans1(u2), d1.size()), d1], dim=1))
        # Final convolution
        output = self.final_conv(u1)
        return output

class UNetSmall(nn.Module):
    def __init__(self, n_classes, in_channels=3, n_filters=16, dropout=0.1, batchnorm=True):
        super(UNetSmall, self).__init__()

        # contracting path
        self.down1 = Conv2DBlock(in_channels, n_filters, batchnorm=batchnorm, dropout=dropout)
        self.down2 = Conv2DBlock(n_filters, n_filters, batchnorm=batchnorm, dropout=dropout)

        self.center = Conv2DBlock(n_filters, n_filters * 2, batchnorm=batchnorm, dropout=dropout)

        # expansive path
        self.trans2 = nn.ConvTranspose2d(n_filters * 2, n_filters * 2, kernel_size=3, stride=2, padding=0)
        self.up2 = Conv2DBlock(n_filters * 2 + n_filters * 2, n_filters * 1, batchnorm=batchnorm, dropout=dropout)
        self.trans1 = nn.ConvTranspose2d(n_filters, n_filters, kernel_size=3, stride=2, padding=0)
        self.up1 = Conv2DBlock(n_filters + n_filters, n_filters, batchnorm=batchnorm, dropout=dropout)

        self.final_conv = nn.Conv2d(n_filters, n_classes, kernel_size=1)

    def forward(self, x):
        # Downsample
        d1 = self.down1(x)
        d2 = self.down2(nn.MaxPool2d(2)(d1))
        c = self.center(nn.MaxPool2d(2)(d2))
        # Upsample
        u2 = self.up2(torch.cat([center_crop(self.trans2(c), d2.size()), d2], dim=1))
        u1 = self.up1(torch.cat([center_crop(self.trans1(u2), d1.size()), d1], dim=1))
        # Final convolution
        output = self.final_conv(u1)
        return output

def get_model_pytorch(model_name, nClasses=1, n_filters=16, dropout=0.1, batchnorm=True, n_channels=3):
    if model_name == 'fcn':
        model = FCN(nClasses, n_filters, dropout, batchnorm)
    elif model_name == 'unet':
        model = UNet(nClasses, n_channels, n_filters, dropout, batchnorm)
    return model