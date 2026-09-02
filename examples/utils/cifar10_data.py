"""Shared CIFAR-10 data helpers for example training workloads."""


def require_torch():
    """torch alone, for the paths that never touch a dataset.

    The synthetic loaders below generate their own tensors, so demanding
    torchvision for them would make `--synthetic-data` - the mode whose whole
    point is running with no dataset - fail on a machine that simply has not
    installed torchvision.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "CIFAR-10 examples require torch. Install it with "
            "`pip install torch` and rerun this script."
        ) from exc

    return torch


def require_torchvision_cifar10():
    try:
        import torch
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError(
            "CIFAR-10 examples require torch and torchvision. Install them with "
            "`pip install torch torchvision` and rerun this script."
        ) from exc

    return torch, DataLoader, datasets, transforms


def build_cifar10_loaders(
    data_dir,
    batch_size,
    download=True,
    max_test_items=10000,
    seed=None,
    shuffle_train=True,
    train_augment=False,
):
    torch, DataLoader, datasets, transforms = require_torchvision_cifar10()
    test_transform = transforms.ToTensor()
    if train_augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        train_transform = test_transform

    train_set = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=train_transform,
    )
    test_set = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=download,
        transform=test_transform,
    )

    if max_test_items and max_test_items < len(test_set):
        test_set = torch.utils.data.Subset(test_set, range(max_test_items))

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle_train,
        generator=generator,
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def build_synthetic_cifar10_loaders(batch_size, train_batches=2, test_batches=1, seed=0):
    torch = require_torch()
    generator = torch.Generator()
    generator.manual_seed(seed)

    def make_batches(num_batches):
        batches = []
        for batch_idx in range(num_batches):
            images = torch.rand(batch_size, 3, 32, 32, generator=generator)
            labels = torch.tensor(
                [(batch_idx * batch_size + i) % 10 for i in range(batch_size)],
                dtype=torch.long,
            )
            batches.append((images, labels))
        return batches

    return make_batches(train_batches), make_batches(test_batches)


def labels_to_list(labels):
    return [int(v) for v in labels.tolist()]
