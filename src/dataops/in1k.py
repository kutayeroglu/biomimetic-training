import os
from typing import Optional

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms


def get_imagenet_dataloaders(
    data_dir,
    val_dir: Optional[str] = None,
    batch_size=128,
    num_workers=8,
    train_frac: Optional[float] = None,
    val_frac: Optional[float] = None,
):
    """Create ImageNet dataloaders with appropriate transforms"""

    # TRAINING: Matches paper text + Keras behavior
    train_transform = transforms.Compose(
        [
            # 1. Resize to explicit 256x256 (Squash) to match "256x256 pixel images" text
            #    and standard Keras flow_from_directory behavior.
            transforms.Resize((256, 256)),
            # 2. Horizontal flipping at random
            #    (Placed here to act on the 256x256 image)
            transforms.RandomHorizontalFlip(p=0.5),
            # 3. Random cropping from 256 to 227
            transforms.RandomCrop(227),
            # 4. Value rescaling [0, 255] to [-1, 1]
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # VALIDATION: Deterministic (Standard Practice)
    val_transform = transforms.Compose(
        [
            # 1. Resize to explicit 256x256
            transforms.Resize((256, 256)),
            # 2. Center Crop (Deterministic) instead of Random Crop
            #    The paper doesn't explicitly specify validation transforms,
            #    but CenterCrop is the standard counterpart to RandomCrop.
            transforms.CenterCrop(227),
            # 3. No Flipping
            # 4. Value rescaling
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"), transform=train_transform
    )

    # Handle validation dataset
    resolved_val_dir = val_dir if val_dir is not None else os.path.join(data_dir, "val")

    if not os.path.isdir(resolved_val_dir):
        raise FileNotFoundError(
            f"Validation directory not found at: {resolved_val_dir}. "
            "Provide a valid path via `val_dir`."
        )

    val_dataset = datasets.ImageFolder(resolved_val_dir, transform=val_transform)

    # Create samplers
    train_sampler = None
    val_sampler = None

    if train_frac is not None:
        num = len(train_dataset)
        # reproducible random split
        g = torch.Generator().manual_seed(42)
        indices = torch.randperm(num, generator=g)[: int(num * train_frac)].tolist()
        train_sampler = SubsetRandomSampler(indices)
    if val_frac is not None:
        num = len(val_dataset)
        g = torch.Generator().manual_seed(43)
        indices = torch.randperm(num, generator=g)[: int(num * val_frac)].tolist()
        val_sampler = SubsetRandomSampler(indices)

    # Create dataloaders
    # NOTE: for pin_memory = True, see: https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, val_loader
