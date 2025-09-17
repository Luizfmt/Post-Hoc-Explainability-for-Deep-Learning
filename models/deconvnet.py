#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# M. D. Zeiler et al. “Visualizing and Understanding Convolutional Networks”. In: ECCV. 2014.
# https://arxiv.org/abs/1311.2901
"""
Deconvolutional NeuralNetwork for adapted AlexNet 28x28 and 128x128
"""

import torch
import torch.nn as nn


class DeconvNet28(nn.Module):
    
    
    """
    Runs the reverse of the AlexNet28 undoing each convolution and maxpooling 
    """
    
    def __init__(self, alexnet28):
        super(DeconvNet28, self).__init__()
        
        # Layer 5
        self.unpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv5 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu5 = nn.ReLU()
        
        # Layer 4
        self.deconv4 = nn.ConvTranspose2d(256, 384, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4 = nn.ReLU()
        
        # Layer 3
        self.deconv3 = nn.ConvTranspose2d(384, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu3 = nn.ReLU()
        
        # Layer 2 
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(192, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.ReLU()
        
        # Layer 1
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv1 = nn.ConvTranspose2d(64, 3, kernel_size=5, stride=1, padding=2, output_padding=1, bias=False)
        self.relu1 = nn.ReLU()
        
        # Copy weights of the conv2d layers of the trained AlexNet28 into the convtranspose2d layers
        
        self.deconv5.weight.data.copy_(alexnet28.conv5.weight.data)
        
        self.deconv4.weight.data.copy_(alexnet28.conv4.weight.data)
        
        self.deconv3.weight.data.copy_(alexnet28.conv3.weight.data)
        
        self.deconv2.weight.data.copy_(alexnet28.conv2.weight.data)
        
        self.deconv1.weight.data.copy_(alexnet28.conv1.weight.data)
        
    def forward(self, x, indices, layer):
        """
        x        : the one‐hot feature map we want to visualize, from layer L
        indices  : a dict { 'idx1':…, 'idx2':…, 'idx5':… } (only idxj that exist are used)
        layer    : integer in {1,2,3,4,5} that tells where to start reversing
        """

        # If layer == 5, we do exactly as before:
        if layer == 5:
            # unpool5:
            x = self.unpool5(x, indices['idx5'], output_size=torch.Size([1,256,7,7]))
            x = self.deconv5(x); x = self.relu5(x)

            # drop through layers 4,3 exactly as before:
            x = self.deconv4(x); x = self.relu4(x)
            x = self.deconv3(x); x = self.relu3(x)

            # unpool2 etc...
            x = self.unpool2(x, indices['idx2'])
            x = self.deconv2(x); x = self.relu2(x)

            x = self.unpool1(x, indices['idx1'])
            x = self.deconv1(x); x = self.relu1(x)
            return x

        # If layer == 4, skip unpool5 and deconv5, because layer 4 was never pooled
        elif layer == 4:
            x = self.deconv4(x) ; x = self.relu4(x)
            x = self.deconv3(x) ; x = self.relu3(x)

            x = self.unpool2(x, indices['idx2'])
            x = self.deconv2(x) ; x = self.relu2(x)

            x = self.unpool1(x, indices['idx1'])
            x = self.deconv1(x) ; x = self.relu1(x)
            return x

        # If layer == 3, skip layers 5,4 entirely:
        elif layer == 3:
            x = self.deconv3(x) ; x = self.relu3(x)

            x = self.unpool2(x, indices['idx2'])
            x = self.deconv2(x) ; x = self.relu2(x)

            x = self.unpool1(x, indices['idx1'])
            x = self.deconv1(x) ; x = self.relu1(x)
            return x

        # If layer == 2, skip everything above layer 2:
        elif layer == 2:
            x = self.unpool2(x, indices['idx2'])
            x = self.deconv2(x) ; x = self.relu2(x)

            x = self.unpool1(x, indices['idx1'])
            x = self.deconv1(x) ; x = self.relu1(x)
            return x

        # If layer == 1, skip all the way to:
        elif layer == 1:
            x = self.unpool1(x, indices['idx1'])
            x = self.deconv1(x) ; x = self.relu1(x)
            return x

        else:
            raise ValueError(f"Invalid layer={layer} (must be 1..5)")


class DeconvNet128(nn.Module):
    
    
    """
    Runs the reverse of the AlexNet28 undoing each convolution and maxpooling 
    """
    
    def __init__(self, alexnet128):
        super(DeconvNet128, self).__init__()
        
        # Layer 5
        self.unpool5 = nn.MaxUnpool2d(kernel_size=3, stride=2)
        self.deconv5 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu5 = nn.ReLU()
        
        # Layer 4
        self.deconv4 = nn.ConvTranspose2d(256, 384, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu4 = nn.ReLU()
        
        # Layer 3
        self.deconv3 = nn.ConvTranspose2d(384, 192, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu3 = nn.ReLU()
        
        # Layer 2 
        self.unpool2 = nn.MaxUnpool2d(kernel_size=3, stride=2)
        self.deconv2 = nn.ConvTranspose2d(192, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.relu2 = nn.ReLU()
        
        # Layer 1
        self.unpool1 = nn.MaxUnpool2d(kernel_size=3, stride=2)
        self.deconv1 = nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.relu1 = nn.ReLU()
        
        # Copy weights of the conv2d layers of the trained AlexNet28 into the convtranspose2d layers
        
        self.deconv5.weight.data.copy_(alexnet128.conv5.weight.data)
        
        self.deconv4.weight.data.copy_(alexnet128.conv4.weight.data)
        
        self.deconv3.weight.data.copy_(alexnet128.conv3.weight.data)
        
        self.deconv2.weight.data.copy_(alexnet128.conv2.weight.data)
        
        self.deconv1.weight.data.copy_(alexnet128.conv1.weight.data)
        
    def forward(self, x, indices, layer):
        """
        x        : the one‐hot feature map we want to visualize, from layer L
        indices  : a dict { 'idx1':…, 'idx2':…, 'idx5':… } (only idxj that exist are used)
        layer    : integer in {1,2,3,4,5} that tells where to start reversing
        """

        # If layer == 5, we do exactly as before:
        if layer == 5:
            # unpool5:
            x = self.unpool5(x, indices['idx5'], output_size=torch.Size([x.size(0),256,15,15]))
            x = self.deconv5(x); #x = self.relu5(x)

            # drop through layers 4,3 exactly as before:
            x = self.deconv4(x); #x = self.relu4(x)
            x = self.deconv3(x); #x = self.relu3(x)

            # unpool2 etc...
            x = self.unpool2(x, indices['idx2'], output_size=torch.Size([ x.size(0), 192, 31, 31 ]))
            x = self.deconv2(x); #x = self.relu2(x)

            x = self.unpool1(x, indices['idx1'], output_size=torch.Size([ x.size(0),  64, 64, 64 ]))
            x = self.deconv1(x); #x = self.relu1(x)
            return x

        # If layer == 4, skip unpool5 and deconv5, because layer 4 was never pooled
        elif layer == 4:
            x = self.deconv4(x) ; #x = self.relu4(x)
            x = self.deconv3(x) ; #x = self.relu3(x)

            x = self.unpool2(x, indices['idx2'], output_size=torch.Size([ x.size(0), 192, 31, 31 ]))
            x = self.deconv2(x) ; #x = self.relu2(x)

            x = self.unpool1(x, indices['idx1'], output_size=torch.Size([ x.size(0),  64, 64, 64 ]))
            x = self.deconv1(x) ; #x = self.relu1(x)
            return x

        # If layer == 3, skip layers 5,4 entirely:
        elif layer == 3:
            x = self.deconv3(x) ; #x = self.relu3(x)

            x = self.unpool2(x, indices['idx2'], output_size=torch.Size([ x.size(0), 192, 31, 31 ]))
            x = self.deconv2(x) ; #x = self.relu2(x)

            x = self.unpool1(x, indices['idx1'], output_size=torch.Size([ x.size(0),  64, 64, 64 ]))
            x = self.deconv1(x) ; #x = self.relu1(x)
            return x

        # If layer == 2, skip everything above layer 2:
        elif layer == 2:
            x = self.unpool2(x, indices['idx2'], output_size=torch.Size([ x.size(0), 192, 31, 31 ]))
            x = self.deconv2(x) ; #x = self.relu2(x)

            x = self.unpool1(x, indices['idx1'], output_size=torch.Size([ x.size(0),  64, 64, 64 ]))
            x = self.deconv1(x) ; #x = self.relu1(x)
            return x

        # If layer == 1, skip all the way to:
        elif layer == 1:
            x = self.unpool1(x, indices['idx1'], output_size=torch.Size([ x.size(0),  64, 64, 64 ]))
            x = self.deconv1(x) ; #x = self.relu1(x)
            return x

        else:
            raise ValueError(f"Invalid layer={layer} (must be 1..5)")
