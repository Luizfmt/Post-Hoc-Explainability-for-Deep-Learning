# -*- coding: utf-8 -*-
"""
GrandCAM implementation

original paper: R. R. Selvaraju et al. “Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based...”. In: IJCV (2020).
"""

import torch
from torch import nn

import torch.nn.functional as F

class GradCam(nn.Module):


        
    """
    Creates heatmaps higliting the relevance of each region in a image 
    to the (decision for a given class)  (improve description) 
    """
    
    def __init__(self, alexnet28):
        """
        Inputs
            alexnet28: Trained alexnet28 neural network object

        """
        super(GradCam, self).__init__()

        self.model = alexnet28



    def _combine_feature_maps(self, activations, grads):
        """
        Gets a final combination of the feature maps (activations),
        weightened by their respective grad images.

        inputs
            activations:    4D tensor of shape [1, k, H, W]
            grads:          4D tensor of shape [1, k, H, W]

        outputs
            heatmap: 4D tensor of shape [1, 1, H, W]
        """
        alphas = grads.mean(dim=(2, 3), keepdim=True) # [1, K, 1, 1]
        weighted_maps = alphas * activations # [1, K, H, W]
        heatmap = weighted_maps.sum(dim=1, keepdim=True) # [1, 1, H, W]
        heatmap = F.relu(heatmap) # Essential
        return heatmap

    def _upsample(self, heatmap, size=(28, 28), interpolation = 'bilinear'):
        """
        Upsamples the combined feature map.
        
        inputs
            heatmap: 4D tensor of shape [1, 1, H, W]
            size: Tuple with the final dimentions
            interpolation: String, indicates the interpolation method

        outputs
            up_image: upsampled image
        """
        
        up_image = F.interpolate(heatmap, size=size, mode=interpolation, align_corners=False)
        return up_image.squeeze(0).squeeze(0)


    def compute_heatmap(self, img, layer, target_class):
        """
        Inputs
            img: pytorch tensor of dimentions 28 x 28(?)
            layer: int from 1 to 5
            target_class: if no valid number is given, the heatmaps of all classes are plotted
            
        Outputs    

        """
        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        if layer == 1:
            target_layer = self.model.conv1
        elif layer == 2:
            target_layer = self.model.conv2
        elif layer == 3:
            target_layer = self.model.conv3
        elif layer == 4:
            target_layer = self.model.conv4
        elif layer == 5:
            target_layer = self.model.conv5


        fwd_handle = target_layer.register_forward_hook(forward_hook)
        bwd_handle = target_layer.register_backward_hook(backward_hook)
        
        

        Y_vector,_ = self.model(img)              # [1, n_classes] (alexNet28 does not finish with softmax)
        
        if not isinstance(target_class, int):
            raise ValueError("Invalid target_class")
        
        Y_c = Y_vector[0, target_class]               # get the specific Y for class c
        
        # get gradients by backward operation
        self.model.zero_grad()
        Y_c.backward()

        # compute and upsample heatmap
        heatmap = self._combine_feature_maps(self._activations, self._gradients)
        heatmap = self._upsample(heatmap)

        # Remove hooks (????)
        fwd_handle.remove()
        bwd_handle.remove()

        return heatmap
