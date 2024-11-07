############################################################
# Modules Loading
############################################################
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from copy import deepcopy
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from torchinfo import summary

############################################################
# extendedMRCNN
############################################################
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class GELU(nn.Module):
    # for older versions of PyTorch.  For new versions you can use nn.GELU() instead.
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        x = torch.nn.functional.gelu(x)
        return x
    
class extendedMRCNN(nn.Module):
    def __init__(self, afr_reduced_cnn_size):
        super(extendedMRCNN, self).__init__()
        drate = 0.5
        fs = 100

        kernel_size1 = int(fs / 0.25)  # delta range
        kernel_size2 = int(fs / 5)      # theta range
        kernel_size3 = int(fs / 10)     # alpha range
        kernel_size4 = int(fs / 20)     # beta range
        kernel_size5 = int(fs / 50)     # gamma range

        def create_feature_layer(kernel_size):
            stride = max(50, round(kernel_size / 8))  # Ensure at least 2
            padding = max(0, round(kernel_size / 2)) #

            return nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=kernel_size, stride=stride, bias=False, padding=padding),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
                nn.Dropout(drate),
                nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
                nn.BatchNorm1d(128),
                nn.GELU(),
                
                nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
            )

        self.features1 = create_feature_layer(kernel_size1)
        self.features2 = create_feature_layer(kernel_size2)
        self.features3 = create_feature_layer(kernel_size3)
        self.features4 = create_feature_layer(kernel_size4)
        self.features5 = create_feature_layer(kernel_size5)

        self.dropout = nn.Dropout(drate)
        self.inplanes = 640
        self.AFR = self._make_layer(SEBasicBlock, afr_reduced_cnn_size, 1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x)
        x3 = self.features3(x)
        x4 = self.features4(x)
        x5 = self.features5(x)
        print(f"x1 shape: {x1.shape}")
        print(f"x2 shape: {x2.shape}")
        print(f"x3 shape: {x3.shape}")
        print(f"x4 shape: {x4.shape}")
        print(f"x5 shape: {x5.shape}")

        perm_feat = 2
        if perm_feat == 1:
            permuted_indices = torch.randperm(x1.size(0))  # Shuffle along batch dimension
            x1_permuted = x1[permuted_indices]
            x_concat = torch.cat((x1_permuted, x2, x3, x4, x5), dim=1)
            print(f"x1 shape: {x1.shape}")

        elif perm_feat == 2:
            permuted_indices = torch.randperm(x2.size(0))  # Shuffle along batch dimension
            x2_permuted = x2[permuted_indices]
            x_concat = torch.cat((x1, x2_permuted, x3, x4, x5), dim=1)
            print(f"x2 shape: {x2.shape}")

        elif perm_feat == 3:
            permuted_indices = torch.randperm(x3.size(0))  # Shuffle along batch dimension
            x3_permuted = x3[permuted_indices]
            x_concat = torch.cat((x1, x2, x3_permuted, x4, x5), dim=1)
            print(f"x3 shape: {x3.shape}")

        elif perm_feat == 4:
            permuted_indices = torch.randperm(x4.size(0))  # Shuffle along batch dimension
            x4_permuted = x4[permuted_indices]
            x_concat = torch.cat((x1, x2, x3, x4_permuted, x5), dim=1)
            print(f"x4 shape: {x4.shape}")

        elif perm_feat == 5:
            permuted_indices = torch.randperm(x5.size(0))  # Shuffle along batch dimension
            x5_permuted = x5[permuted_indices]
            x_concat = torch.cat((x1, x2, x3, x4, x5_permuted), dim=1)
            print(f"x5 shape: {x5.shape}")
        
        # Concatenate all feature outputs along the channel dimension
        # x_concat = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x_concat = self.dropout(x_concat)
        x_concat = self.AFR(x_concat)

        print(f'shape of extendedMRCNN output{x_concat.shape}')

        return x_concat
    
############################################################
# temporal context encoder
############################################################
class TContext(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(TContext, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)
  def forward(self, x):
      lstm_out, _ = self.lstm(x)
      out = self.fc(lstm_out[:, -1, :])
      return F.log_softmax(out, dim=1)
############################################################
# completed model
############################################################
class DeepSleepSEEG(nn.Module):
  def __init__(self):
        super(DeepSleepSEEG, self).__init__()
        num_classes = 4
        afr_reduced_cnn_size = 640
        self.extendedMRCNN = extendedMRCNN(afr_reduced_cnn_size)
        lstm_input_size = 9 # Output size from extendedMRCNN
        hidden_size = 128  # Number of LSTM units
        num_layers = 2     # Number of LSTM layers

        # Initialize TContext with appropriate parameters
        self.TContext = TContext(input_size=lstm_input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes)

  def forward(self, x):
      # Get features from extendedMRCNN
      x_feat = self.extendedMRCNN(x)
      # Pass features through TContext (LSTM)
      encoded_features = self.TContext(x_feat)

      return encoded_features
