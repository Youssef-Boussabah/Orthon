"""Shared MNIST data helpers for example training workloads."""


def require_torchvision_mnist():
    try:
        import torch
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError(
            "MNIST examples require torch and torchvision. Install them with "
            "`pip install torch torchvision` and rerun this script."
        ) from exc

    return torch, DataLoader, datasets, transforms


def build_mnist_loaders(data_dir, batch_size, download=True, max_test_items=10000):
    torch, DataLoader, datasets, transforms = require_torchvision_mnist()
    transform = transforms.ToTensor()

    train_set = datasets.MNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=transform,
    )
    test_set = datasets.MNIST(
        root=data_dir,
        train=False,
        download=download,
        transform=transform,
    )

    if max_test_items and max_test_items < len(test_set):
        test_set = torch.utils.data.Subset(test_set, range(max_test_items))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def flatten_images(images):
    return images.reshape(images.shape[0], 784).float()


def labels_to_list(labels):
    return [int(v) for v in labels.tolist()]
