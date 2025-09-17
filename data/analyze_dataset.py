import torch
from collections import Counter
import matplotlib.pyplot as plt


def get_class_distribution(dataloader):
    """
    Counts how many samples there are per class in the dataloader.

    Args:
        dataloader (DataLoader): loader with labeled dataset

    Returns:
        dict: count of examples per class
    """
    
    all_labels = []
    for _, labels in dataloader:
        if labels.ndim > 1:                 # checks if labels are not already 1D and adjusts 
            if labels.shape[1] == 1:
                labels = labels.squeeze(1) 
            else:
                labels = labels.argmax(dim=1)  
        all_labels.extend(labels.tolist()) # add label values to the list
    # Count occurrences of each class
    return Counter(all_labels)


def plot_class_distribution(counter_dict, class_names=None):
    """
    Plots a bar chart showing the number of samples per class.


    Args:
        counter_dict (dict): dictionary mapping class indices to sample counts 
        class_names (dict or list or None): class labels to display on the x-axis.
                                            If None, class indices will be used instead.
    """
    # Sort the dictionary by class index 
    sorted_items = sorted(counter_dict.items())

    # Extract class indices and their corresponding sample counts
    keys = [k for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    # Convert indices to class names if provided
    labels = [class_names[k] if class_names else str(k) for k in keys]

    # Print the relation between index and class name
    if class_names:
        print("Class index and name:")
        for k in keys:
            print(f"{k}: {class_names[k]}")
        print()

    # Create the bar chart
    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values, color='lightseagreen')

    # Each bar with its value (sample count)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 300, str(yval),
                 ha='center', va='bottom', fontsize=9)

    # Set chart labels and title
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.title("PathMNIST - Class Distribution")

    # Rotate x-axis labels if class names are long
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.show()


def compute_mean_std(dataloader):
    """
    Estimates the mean and standard deviation of the pixels in the dataset.

    Args:
        dataloader (DataLoader): loader with images

    Returns:
        tuple: mean and standard deviation (float)
    """
    mean = 0.0
    std = 0.0
    total = 0
    for images, _ in dataloader:
        batch_samples = images.size(0)            # number of images in this batch
        images = images.view(batch_samples, -1)   
        # Compute mean and std per-image and sum over the batch
        mean += images.mean(1).sum()              
        std += images.std(1).sum()                
        total += batch_samples    
     # Return the global average mean and std across all images                
    return mean / total, std / total
