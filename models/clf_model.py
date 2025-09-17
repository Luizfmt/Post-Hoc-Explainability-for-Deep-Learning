# AlexNet ( Krizhevsky, A., Sutskever, I., and Hinton, G.E. Imagenet classification with deep convolutional neural )
# https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

"""
Adaptation of AlexNet for different input sizes

28x28 and 128x128

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data

# define pytorch device - useful for device-agnostic execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AlexNet28(nn.Module):
    
    """
    Modified AlexNet to support 28x28 inputs of medmnist dataset and to return 
    the indices of each MaxPool2d layer. 
    
    The forward method returns both the final logits and a dict of intermediate activations.
    """
    
    def __init__(self, num_classes=9):
        super(AlexNet28,self).__init__()
        
        # Layer 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2) # 28x28
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        # Layer 2
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1) # 14x14
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True) # 7x7

        # Layer 3
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1) # 7x7
        self.relu3 = nn.ReLU()
        
        # Layer 4
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1) # 7x7
        self.relu4 = nn.ReLU()
        
        # Layer 5
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1) # 7x7
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True) # 3x3
        
        # -----------------------------------------------------------------------------
        # Classifier
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256*3*3, 4096)
        self.relu_fc1 = nn.ReLU()
        
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.relu_fc2 = nn.ReLU()
        
        self.fc3 = nn.Linear(4096, num_classes)
        
    def forward(self, x):
        """
        Returns: x (tensor of shape [batch, num_classes])
                 acts : Dict containing features and indices after each maxpooling
        """
        
        acts ={}
        
        # Layer 1 : conv1, relu1, pool1
        x = self.conv1(x)
        x = self.relu1(x)
        x, idx1 = self.pool1(x)
        
        # store feature (feat1) and indice (idx1)
        acts['feat1'] = x.clone() # [batch_size, 64, 14, 14]
        acts['idx1'] = idx1 # [batch_size, 64, 14, 14]

        # Layer 2 : conv2, relu2, pool2
        x = self.conv2(x)
        x = self.relu2(x)
        x, idx2 = self.pool2(x)

        # store
        acts['feat2'] = x.clone() # [batch_size, 192, 7, 7]
        acts['idx2'] = idx2 # [batch_size, 192, 7, 7]
        
        # Layer 3 : conv3, relu3
        x = self.conv3(x)
        x = self.relu3(x)
        acts['feat3'] = x.clone()
        
        # Layer 4 : conv4, relu 4
        x = self.conv4(x)
        x = self.relu4(x)
        acts['feat4'] = x.clone()
        
        # Layer 5: conv5, relu5, pool5
        x = self.conv5(x)
        x = self.relu5(x)
        x, idx5 = self.pool5(x)
        
        # store
        acts['feat5'] = x.clone() # [batch_size, 256, 3, 3]
        acts['idx5'] = idx5 # [batch_size, 256, 3, 3]
        
        # Flatten for the classifier part
        x = x.view(x.size(0),-1)
        
        # Classifier layers : fc1
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        
        # fc2
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)
        
        #fc3
        x = self.fc3(x)

        return x, acts


class AlexNet128(nn.Module):
    
    """
    Modified AlexNet to support 128x128 inputs of medmnist dataset and to return 
    the indices of each MaxPool2d layer. 
    
    The forward method returns both the final logits and a dict of intermediate activations.
    """
    
    def __init__(self, num_classes=9):
        super(AlexNet128,self).__init__()
        
        # Layer 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2) # 128x128 to 64x64
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, return_indices=True) # 64x64 to 31x31
        
        # Layer 2
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2) # remains 31x31 
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, return_indices=True) # 31x31 to 15x15

        # Layer 3
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1) # remains 15x15
        self.relu3 = nn.ReLU()
        
        # Layer 4
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1) # remains 15x15
        self.relu4 = nn.ReLU()
        
        # Layer 5
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1) # remains 15x15
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2, return_indices=True) # 15x15 to 7x7
        
        # Force dimensions to be 6x6
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # -----------------------------------------------------------------------------
        # Classifier
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256*6*6, 4096)
        self.relu_fc1 = nn.ReLU()
        
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.relu_fc2 = nn.ReLU()
        
        self.fc3 = nn.Linear(4096, num_classes)
        
    def forward(self, x):
        """
        Returns: x (tensor of shape [batch, num_classes])
                 acts : Dict containing features and indices after each maxpooling
        """
        
        acts ={}
        
        # Layer 1 : conv1, relu1, pool1
        x = self.conv1(x)
        x = self.relu1(x)
        x, idx1 = self.pool1(x)
        
        # store feature (feat1) and indice (idx1)
        acts['feat1'] = x.clone() # [batch_size, 64, 14, 14]
        acts['idx1'] = idx1 # [batch_size, 64, 14, 14]

        # Layer 2 : conv2, relu2, pool2
        x = self.conv2(x)
        x = self.relu2(x)
        x, idx2 = self.pool2(x)

        # store
        acts['feat2'] = x.clone() # [batch_size, 192, 7, 7]
        acts['idx2'] = idx2 # [batch_size, 192, 7, 7]
        
        # Layer 3 : conv3, relu3
        x = self.conv3(x)
        x = self.relu3(x)
        acts['feat3'] = x.clone()
        
        # Layer 4 : conv4, relu 4
        x = self.conv4(x)
        x = self.relu4(x)
        acts['feat4'] = x.clone()
        
        # Layer 5: conv5, relu5, pool5
        x = self.conv5(x)
        x = self.relu5(x)
        x, idx5 = self.pool5(x)
        
        # store
        acts['feat5'] = x.clone() # [batch_size, 256, 3, 3]
        acts['idx5'] = idx5 # [batch_size, 256, 3, 3]
        
        x = self.avgpool(x)
        
        # Flatten for the classifier part
        x = x.view(x.size(0),-1)
        
        # Classifier layers : fc1
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu_fc1(x)
        
        # fc2
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.relu_fc2(x)
        
        #fc3
        x = self.fc3(x)

        return x, acts

