import os
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


def apply_regimen_transforms(images, blur_sigma, grayscale):
    """Applies on-the-fly transforms for biomimetic regimens."""
    if grayscale:
        images = TF.rgb_to_grayscale(images, num_output_channels=3)
    if blur_sigma > 0:
        # Kernel size is typically 4 * sigma + 1 to capture the Gaussian spread
        kernel_size = int(2 * int(2 * blur_sigma + 0.5) + 1)
        images = TF.gaussian_blur(
            images,
            kernel_size=[kernel_size, kernel_size],
            sigma=[blur_sigma, blur_sigma],
        )
    return images


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    total_epochs: int,
    start_epoch: int = 0,
    best_val_acc: float = 0.0,
    transition_epoch: int = 0,
    phase1_blur_sigma: float = 0.0,
    phase1_grayscale: bool = False,
    phase2_blur_sigma: float = 0.0,
    phase2_grayscale: bool = False,
    device: str = "cuda",
    save_path: str = "checkpoint.pth",
    use_wandb: bool = True,
    wandb_project: str = "biomimetic-training",
    wandb_run_name: str = None,
) -> None:
    # Determine training regimen name for W&B
    if transition_epoch == 0:
        regimen = "Standard"
    elif phase1_blur_sigma > 0 or phase1_grayscale:
        regimen = (
            "Biomimetic"
            if (phase2_blur_sigma == 0 and not phase2_grayscale)
            else "Custom"
        )
    elif phase2_blur_sigma > 0 and phase2_grayscale:
        regimen = "Anti-Biomimetic"
    else:
        regimen = "Standard"

    # Initialize W&B
    if use_wandb and WANDB_AVAILABLE:
        if wandb_run_name is None:
            wandb_run_name = f"{regimen.lower()}_{total_epochs}ep"

        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "regimen": regimen,
                "total_epochs": total_epochs,
                "transition_epoch": transition_epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "model": type(model).__name__,
            },
        )

    best_val_acc = best_val_acc
    best_checkpoint_path = None  # Track the current best checkpoint filename

    for epoch in range(start_epoch, total_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Determine current phase parameters
        if epoch < transition_epoch:
            curr_sigma, curr_gray = phase1_blur_sigma, phase1_grayscale
        else:
            curr_sigma, curr_gray = phase2_blur_sigma, phase2_grayscale

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs} [{regimen}]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Apply Phase-specific augmentations
            images = apply_regimen_transforms(images, curr_sigma, curr_gray)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(
                {"Loss": f"{loss.item():.4f}", "Acc": f"{100.0 * correct / total:.2f}%"}
            )

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = 100.0 * val_correct / val_total

        print(
            f"Summary: Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Val Acc: {val_epoch_acc:.2f}%"
        )

        # Log metrics to W&B
        if use_wandb and WANDB_AVAILABLE:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": epoch_loss,
                    "train_acc": epoch_acc,
                    "val_loss": val_epoch_loss,
                    "val_acc": val_epoch_acc,
                    "phase_sigma": curr_sigma,
                    "phase_grayscale": int(curr_gray),
                }
            )

        # Prepare checkpoint data
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_epoch_acc,
        }

        # Determine checkpoint filename with epoch number
        base_path, ext = os.path.splitext(save_path)
        checkpoint_path = f"{base_path}_epoch{epoch + 1}{ext}"

        # Save checkpoint at transition epoch
        if transition_epoch > 0 and epoch == transition_epoch:
            torch.save(checkpoint_data, checkpoint_path)
            print(f"Transition epoch checkpoint saved to {checkpoint_path}")

        # Save latest checkpoint periodically for resuming (every 10 epochs)
        if (epoch + 1) % 10 == 0 or epoch + 1 == total_epochs:
            torch.save(checkpoint_data, save_path)
            print(f"Latest checkpoint saved to {save_path} (Epoch {epoch + 1})")

        # Save best validation accuracy checkpoint (overwrites previous best)
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            # Delete previous best checkpoint file if it exists
            if best_checkpoint_path is not None and os.path.exists(
                best_checkpoint_path
            ):
                os.remove(best_checkpoint_path)
            # Save new best checkpoint with epoch number
            best_checkpoint_path = f"{base_path}_best_epoch{epoch + 1}{ext}"
            torch.save(checkpoint_data, best_checkpoint_path)
            print(
                f"Best checkpoint saved to {best_checkpoint_path} (Val Acc: {val_epoch_acc:.2f}%)"
            )

    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
