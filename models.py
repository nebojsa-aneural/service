import numpy as np
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryJaccardIndex

device = ('cuda' if torch.cuda.is_available() else 'cpu')

ce = nn.CrossEntropyLoss()

def UnetLoss(preds, targets):
    ce_loss = ce(preds, targets)
    return ce_loss

def UnetAccuracy(preds, targets):
    metric = BinaryJaccardIndex().to(device)
    return metric(preds, targets)

def combine_outputs(outputs):
    list_of_tensors = []
    for pred_mask in outputs:
        pred_mask = pred_mask.permute(1,2,0).cpu().detach().numpy()
        firstChannelMask = pred_mask[:,:,0]
        firstChannelMask[firstChannelMask <= firstChannelMask.mean()] = 0
        firstChannelMask[firstChannelMask > firstChannelMask.mean()] = 1

        secondChannelMask = pred_mask[:,:,1]
        secondChannelMask[secondChannelMask <= secondChannelMask.mean()] = 0
        secondChannelMask[secondChannelMask > secondChannelMask.mean()] = 1

        mask_pred = 1 - (firstChannelMask - secondChannelMask)
        threshold = mask_pred.mean()

        mask_pred[mask_pred <= threshold] = 0
        mask_pred[mask_pred > threshold] =1
        mask_pred.astype(np.int8)

        list_of_tensors.append(torch.from_numpy(mask_pred))

    return torch.stack(list_of_tensors)

class UNet(nn.Module):
    def __init__(self, input_size=(256,256,1)):
        super(UNet, self).__init__()
        
        self.input_size = input_size
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU())
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU())
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU())
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(256),
                                  nn.ReLU())
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(512),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(512),
                                  nn.ReLU())

        self.up6 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.conv6 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(256),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(256),
                                  nn.ReLU())

        self.up7 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv7 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(128),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(128),
                                  nn.ReLU())

        self.up8 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.conv8 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(64),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(64),
                                  nn.ReLU())

        self.up9 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.conv9 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(32),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(32),
                                  nn.ReLU())

        self.conv10 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.conv5(pool4)

        up6 = self.up6(conv5)
        up6 = torch.cat((up6, conv4), dim=1)
        conv6 = self.conv6(up6)

        up7 = self.up7(conv6)
        up7 = torch.cat((up7, conv3), dim=1)
        conv7 = self.conv7(up7)

        up8 = self.up8(conv7)
        up8 = torch.cat((up8, conv2), dim=1)
        conv8 = self.conv8(up8)

        up9 = self.up9(conv8)
        up9 = torch.cat((up9, conv1), dim=1)
        conv9 = self.conv9(up9)
        output = self.conv10(conv9)

        return output