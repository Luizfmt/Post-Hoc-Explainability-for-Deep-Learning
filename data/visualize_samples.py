import matplotlib.pyplot as plt
import torchvision
import torch

def visualize_batch(dataloader, class_names=None, n=8):
    """
    Plots a grid with 'n' images from a batch of dataloader.

    Args:
    dataloader (DataLoader): PyTorch dataloader (e.g. train_loader)
    class_names (list[str] or None): class names, if you want to show
    n (int): number of images to show
    """
    images, labels = next(iter(dataloader))
    images = images[:n]
    labels = labels[:n]

    grid = torchvision.utils.make_grid(images, nrow=n, normalize=True, pad_value=1)
    plt.figure(figsize=(n * 2, 2))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis('off')

    if class_names:
        print("Labels:", [class_names[label.item()] for label in labels])
    else:
        print("Labels (raw):", labels.tolist())

    plt.show()

def visualize_one_sample_per_class(dataloader, class_names=None):
    """
    Displays one sample image for each class in a 3x3 grid.

    Args:
        dataloader (DataLoader): PyTorch dataloader (e.g., train_loader)
        class_names (dict or list or None): optional mapping from class index to class name
    """
    seen_classes = set()
    samples = dict()

    # Collect one sample per class
    for images, labels in dataloader:
        if labels.ndim > 1:
            if labels.shape[1] == 1:
                labels = labels.squeeze(1)
            else:
                labels = labels.argmax(dim=1)

        for img, label in zip(images, labels):
            label_id = label.item()
            if label_id not in seen_classes:
                samples[label_id] = img
                seen_classes.add(label_id)
            if len(seen_classes) >= len(set(labels.tolist())):
                break
        if len(seen_classes) >= len(set(labels.tolist())):
            break

    # Sort and prepare for plotting
    sorted_keys = sorted(samples.keys())
    images = [samples[k] for k in sorted_keys]
    titles = [class_names[k] if class_names else str(k) for k in sorted_keys]

    # Plot as 3x3 grid
    plt.figure(figsize=(9, 9))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(3, 3, i + 1)
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  # Normalize for display
        plt.imshow(img_np)
        plt.title(title, fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_one_sample_per_class_single_row(dataloader, class_names=None):
    """
    Displays one sample image for each class in a 1x9 row.

    Args:
        dataloader (DataLoader): PyTorch dataloader (e.g., train_loader)
        class_names (dict or list or None): optional mapping from class index to class name
    """
    seen_classes = set()
    samples = dict()

    # Collect one sample per class
    for images, labels in dataloader:
        if labels.ndim > 1:
            if labels.shape[1] == 1:
                labels = labels.squeeze(1)
            else:
                labels = labels.argmax(dim=1)

        for img, label in zip(images, labels):
            label_id = label.item()
            if label_id not in seen_classes:
                samples[label_id] = img
                seen_classes.add(label_id)
            if len(seen_classes) >= len(set(labels.tolist())):
                break
        if len(seen_classes) >= len(set(labels.tolist())):
            break

    # Sort and prepare for plotting
    sorted_keys = sorted(samples.keys())
    images = [samples[k] for k in sorted_keys]
    titles = [class_names[k] if class_names else str(k) for k in sorted_keys]

    # Plot as 1x9 row
    plt.figure(figsize=(18, 2.5))  # wider figure for 9 horizontal plots
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 9, i + 1)
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  # Normalize
        plt.imshow(img_np)
        plt.title(title, fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

