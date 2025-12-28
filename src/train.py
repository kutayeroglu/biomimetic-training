import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


def train_model(
    # 1. Standard Training Boilerplate
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    total_epochs: int,
    # 2. Regimen Control Parameters (Phase 1 -> Phase 2)
    transition_epoch: int = 0,  # Epoch to switch from Phase 1 to Phase 2
    # Phase 1 Parameters (Before Transition)
    phase1_blur_sigma: float = 0.0,  # Sigma for Gaussian blur in Phase 1
    phase1_grayscale: bool = False,  # Whether to use grayscale in Phase 1
    # Phase 2 Parameters (After Transition)
    phase2_blur_sigma: float = 0.0,  # Sigma for Gaussian blur in Phase 2
    phase2_grayscale: bool = False,  # Whether to use grayscale in Phase 2
    # 3. Hardware/Misc
    device: str = "cuda",
    save_path: str = "checkpoint.pth",
    # 4. Logging
    use_wandb: bool = True,
    wandb_project: str = "biomimetic-training",
    wandb_run_name: str = None,
) -> None:
    """
    Main training loop supporting Standard, Biomimetic, and Anti-Biomimetic regimens.

    Logic:
        if current_epoch < transition_epoch:
            Apply Phase 1 settings (phase1_blur_sigma, phase1_grayscale)
        else:
            Apply Phase 2 settings (phase2_blur_sigma, phase2_grayscale)
    """
    # Determine training regimen
    if transition_epoch == 0:
        regimen = "Standard"
        print("\nTraining Regimen: Standard (constant high-quality input)")
    elif phase1_blur_sigma > 0 or phase1_grayscale:
        if phase2_blur_sigma == 0 and not phase2_grayscale:
            regimen = "Biomimetic"
            print("\nTraining Regimen: Biomimetic (degraded → clear)")
        else:
            regimen = "Custom"
            print("\nTraining Regimen: Custom")
    elif phase2_blur_sigma > 0 or phase2_grayscale:
        regimen = "Anti-Biomimetic"
        print("\nTraining Regimen: Anti-Biomimetic (clear → degraded)")
    else:
        regimen = "Standard"
        print("\nTraining Regimen: Standard (constant high-quality input)")

    # Initialize W&B
    if use_wandb and WANDB_AVAILABLE:
        # Create run name if not provided
        if wandb_run_name is None:
            wandb_run_name = f"{regimen.lower()}_{transition_epoch}ep"
            if phase1_blur_sigma > 0 or phase1_grayscale:
                wandb_run_name += f"_p1b{phase1_blur_sigma}g{int(phase1_grayscale)}"
            if phase2_blur_sigma > 0 or phase2_grayscale:
                wandb_run_name += f"_p2b{phase2_blur_sigma}g{int(phase2_grayscale)}"

        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                # Training regimen
                "regimen": regimen,
                "total_epochs": total_epochs,
                "transition_epoch": transition_epoch,
                "phase1_blur_sigma": phase1_blur_sigma,
                "phase1_grayscale": phase1_grayscale,
                "phase2_blur_sigma": phase2_blur_sigma,
                "phase2_grayscale": phase2_grayscale,
                # Optimizer config (extract from optimizer)
                "optimizer": type(optimizer).__name__,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "momentum": optimizer.param_groups[0].get("momentum", 0),
                "nesterov": optimizer.param_groups[0].get("nesterov", False),
                # Data config
                "batch_size": train_loader.batch_size,
                "train_batches": len(train_loader),
                "val_batches": len(val_loader),
                # Model config
                "model": type(model).__name__,
                "num_params": sum(p.numel() for p in model.parameters()),
                # Hardware
                "device": device,
                # Paths
                "save_path": save_path,
            },
        )
        print(f"W&B initialized: {wandb.run.url}")
    elif use_wandb and not WANDB_AVAILABLE:
        print(
            "Warning: W&B requested but not installed. Install with: pip install wandb"
        )
    else:
        print("W&B logging disabled")

    # TODO: Implement training loop
    # During training, log metrics like:
    #   wandb.log({
    #       "epoch": epoch,
    #       "train_loss": train_loss,
    #       "train_acc": train_acc,
    #       "val_loss": val_loss,
    #       "val_acc": val_acc,
    #       "learning_rate": optimizer.param_groups[0]["lr"],
    #   })

    # At the end, optionally log model artifact:
    #   wandb.log_artifact(save_path, type="model")

    pass


# 1. Standard Regimen (Control) Constant high-quality input.
# train_model(..., transition_epoch=0,
#             phase1_blur_sigma=0, phase1_grayscale=False,  # Initial: Clear
#             phase2_blur_sigma=0, phase2_grayscale=False)  # Final: Clear

# TODO : double-check sigmas
# 2. Biomimetic Regimen (Developmental) Starts degraded (Phase 1), improves to clear (Phase 2).
# train_model(..., transition_epoch=100,
#             phase1_blur_sigma=4.0, phase1_grayscale=True,   # Initial: Blurry/Gray
#             phase2_blur_sigma=0.0, phase2_grayscale=False)  # Final: Clear/Color


# 3. Anti-Biomimetic Regimen (Reverse) Starts clear (Phase 1), degrades to blurry (Phase 2).
# train_model(..., transition_epoch=100,
#             phase1_blur_sigma=0.0, phase1_grayscale=False,  # Initial: Clear/Color
#             phase2_blur_sigma=4.0, phase2_grayscale=True)   # Final: Blurry/Gray
