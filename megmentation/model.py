import torch.nn as nn
import torch
import torchvision.models as models

class TransferLearningSegmentationModel(nn.Module):
    def __init__(self, num_classes):
        super(TransferLearningSegmentationModel, self).__init__()

        # Load a pre-trained ResNet model and remove the fully connected layer
        resnet = models.resnet34(pretrained=True)
        self.encoder1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1
        )
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        self.upconv3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.upconv0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder0 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        b = self.bottleneck(e4)

        u3 = self.upconv3(b)
        e4_upsampled = nn.functional.interpolate(e4, size=(u3.size(2), u3.size(3)), mode='bilinear', align_corners=True)
        d3 = torch.cat((u3, e4_upsampled), dim=1)
        d3 = self.decoder3(d3)
        
        u2 = self.upconv2(d3)
        e3_upsampled = nn.functional.interpolate(e3, size=(u2.size(2), u2.size(3)), mode='bilinear', align_corners=True)
        d2 = torch.cat((u2, e3_upsampled), dim=1)
        d2 = self.decoder2(d2)
        
        u1 = self.upconv1(d2)
        e2_upsampled = nn.functional.interpolate(e2, size=(u1.size(2), u1.size(3)), mode='bilinear', align_corners=True)
        d1 = torch.cat((u1, e2_upsampled), dim=1)
        d1 = self.decoder1(d1)
        
        u0 = self.upconv0(d1)
        e1_upsampled = nn.functional.interpolate(e1, size=(u0.size(2), u0.size(3)), mode='bilinear', align_corners=True)
        d0 = torch.cat((u0, e1_upsampled), dim=1)
        d0 = self.decoder0(d0)

        out = self.final_conv(d0)

        # Resize the output to match the target mask size
        out = nn.functional.interpolate(out, size=(240, 320), mode='bilinear', align_corners=True)
        
        return out
