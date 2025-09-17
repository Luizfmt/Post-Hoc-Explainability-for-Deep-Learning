'''
Functions for occlusion model 
'''

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image


import matplotlib.pyplot as plt
import numpy as np


def get_occlusion_result(img, label, x, y, patch_size, clf_model, layer):
    """
    Applies grey square occlusion and returns relevant values.
    """
    occluded = img.clone()                                # copy input image
    occluded[:, :, y:y+patch_size, x:x+patch_size] = 0.5  # apply the grey square

    with torch.no_grad():
        logits_occ, acts_occ = clf_model(occluded)                  # forward pass the occluded img through the model
        feat_occ = acts_occ[f"feat{layer}"]                             # extract feature map activations from layer 5
        prob_occ = F.softmax(logits_occ, dim=1)[0, label].item()  # compute probability of true class with softmax
        pred_class = logits_occ.argmax(dim=1).item()              # get the predicted class
        # compute the total activation of the strongest feature map in layer 5
        activation = feat_occ[0, strongest_idx].sum().item()    

    # Return:
    # the occluded image (converted to shape [3, H, W] and moved to CPU)
    # the total activation value of the strongest feature
    # the probability of the true class
    # the predicted class index
    return occluded.squeeze().cpu(), activation, prob_occ, pred_class

def plot_occlusion_entry(occluded_img, activation, prob, pred_class, recon_np, x, y):
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))

    axs[0].imshow(to_pil_image(occluded_img))
    axs[0].set_title(f"(a) Patch at ({x},{y})")
    axs[0].axis("off")

    axs[1].imshow(np.full((7, 7), activation), cmap="hot")
    axs[1].set_title(f"(b) Layer 5 activation\nsum = {activation:.2f}")
    axs[1].axis("off")

    axs[2].imshow(recon_np)
    axs[2].set_title(f"(c) Feature projection")
    axs[2].axis("off")

    axs[3].imshow(np.full((7, 7), prob), cmap="coolwarm", vmin=0, vmax=1)
    axs[3].set_title(f"(d) P(true class) = {prob:.2f}")
    axs[3].axis("off")

    pred_name = class_names.get(pred_class, str(pred_class))
    axs[4].imshow(np.full((7, 7), pred_class), cmap="tab20")
    axs[4].set_title(f"(e) Predicted class:\n{pred_name}")
    axs[4].axis("off")

    plt.tight_layout()
    plt.show()

