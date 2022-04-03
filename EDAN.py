import torch
import torch.nn as nn
import ResNet
import emmd

class EDAN(nn.Module):

    def __init__(self, num_classes=31, bottle_neck=True):
        super(EDAN, self).__init__()
        self.feature_layers = ResNet.resnet50(True)
        self.emmd_loss = emmd.EMMD_loss(class_num=num_classes)
        self.bottle_neck = bottle_neck
        if bottle_neck:
            # self.bottle = nn.Linear(2048, 256)
            self.bottle = nn.Sequential(
                nn.Linear(2048, 256),
                nn.BatchNorm1d(256),
                nn.ReLU()
            )
            self.cls_fc = nn.Linear(256, num_classes)
        else:
            self.cls_fc = nn.Linear(2048, num_classes)

    def forward(self, source, target, s_label):
        source = self.feature_layers(source)
        if self.bottle_neck:
            source = self.bottle(source)
        s_pred = self.cls_fc(source)
        target = self.feature_layers(target)
        if self.bottle_neck:
            target = self.bottle(target)
        t_label = self.cls_fc(target)
        loss_emmd = self.emmd_loss.get_loss(source, target, s_label,
                                            s_pred, t_label)
        return s_pred, loss_emmd

    def predict(self, x):
        x = self.feature_layers(x)
        if self.bottle_neck:
            x = self.bottle(x)
        return self.cls_fc(x)

    def get_parameters(self, base_lr=1.0):
        params = [
            {"params": self.feature_layers.parameters(), "lr": 0.1 * base_lr},
            {"params": self.bottle.parameters(), "lr": 1.0 * base_lr},
            {"params": self.cls_fc.parameters(), "lr": 1.0 * base_lr},
        ]

        return params