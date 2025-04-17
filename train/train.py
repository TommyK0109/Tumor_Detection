import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from models.UNetResNet import UNetResNet3D

class BraTSDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_files = sorted(glob.glob(os.path.join(image_dir, "*.npy")))
        self.mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.npy")))
        assert len(self.image_files) == len(self.mask_files), "Mismatch between images and masks"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = np.load(self.image_files[idx])
        mask = np.load(self.mask_files[idx])

        image = torch.tensor(image, dtype=torch.float32).permute(3, 0, 1, 2)
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask

# Function to load checkpoint
def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    if os.path.exists(filename):
        checkpoint = torch.load(filename, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        num_epochs = checkpoint['num_epochs']
        best_dice = checkpoint['best_dice']
        train_losses = checkpoint['train_losses']
        dice_scores = checkpoint['dice_scores']
        print(f"Loaded checkpoint from {filename}. Resuming from epoch {start_epoch} with best Dice: {best_dice:.4f}")
        return start_epoch, num_epochs, best_dice, train_losses, dice_scores
    print(f"No checkpoint found at {filename}. Starting from scratch...")
    return 0, 0, 0.0, [], []

# Function to load best model (if exists)
def load_best_model(model, filename="model_trained.pth"):
    if os.path.exists(filename):
        checkpoint = torch.load(filename, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_dice = checkpoint['best_dice']
        print(f"Loaded best model from {filename} with Dice Score: {best_dice:.4f}")
        return best_dice
    return 0.0

def main():
    # Set paths
    image_path = r"A:\Testing\data\BraTS2020_Preprocessed\images"
    mask_path = r"A:\Testing\data\BraTS2020_Preprocessed\masks"

    # Create Dataset & DataLoader
    train_dataset = BraTSDataset(image_path, mask_path)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
    print(f"Number of training samples: {len(train_dataset)}")

    # Check device
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load model
    model = UNetResNet3D(resnet_type="resnet50_3d", in_channels=4, out_channels=4).to(device)

    # Define loss functions
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(to_onehot_y=True, softmax=True)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Load checkpoint for resuming training (if exists)
    start_epoch, num_epochs, best_dice, train_losses, dice_scores = load_checkpoint(model, optimizer)

    # Load best model (if exists) to initialize best_dice
    best_dice_from_model = load_best_model(model)
    best_dice = max(best_dice, best_dice_from_model)

    # If starting from scratch, set num_epochs
    if start_epoch == 0:
        num_epochs = 2  # Default number of epochs for a fresh run
    total_epochs = start_epoch + 2  # Adjust based on how many more epochs you want

    # Training loop
    try:
        for epoch in range(start_epoch, total_epochs):
            model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch_idx, batch in enumerate(train_loader):
                images, masks = batch
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = model(images)

                # Add channel dimension to masks for DiceLoss
                masks_with_channel = masks.unsqueeze(1)

                # Compute losses
                loss_ce = ce_loss(outputs, masks)
                loss_dice = dice_loss(outputs, masks_with_channel)
                loss = loss_ce + loss_dice

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # Compute accuracy (voxel-wise)
                predicted = torch.argmax(outputs, dim=1)
                correct = (predicted == masks).sum().item()
                total = masks.numel()
                epoch_correct += correct
                epoch_total += total

            avg_loss = epoch_loss / len(train_loader)
            accuracy = epoch_correct / epoch_total
            train_losses.append(avg_loss)
            print(f"Epoch [{epoch + 1}/{total_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Dice: {dice_scores[-1] if dice_scores else 'N/A'}")

            model.eval()
            dice_metric.reset()
            with torch.no_grad():
                for batch in train_loader:
                    images, masks = batch
                    images, masks = images.to(device), masks.to(device)

                    outputs = model(images)
                    outputs = torch.argmax(outputs, dim=1)
                    dice_metric(outputs, masks)

                mean_dice = dice_metric.aggregate().item()
                dice_scores.append(mean_dice)
                print(f"Validation Dice Score: {mean_dice:.4f}")

                # Save checkpoint every epoch (for resuming)
                checkpoint = {
                    'epoch': epoch + 1,
                    'num_epochs': total_epochs,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_dice': best_dice,
                    'train_losses': train_losses,
                    'dice_scores': dice_scores
                }
                torch.save(checkpoint, "checkpoint.pth")
                print("Checkpoint saved for resuming training")

                # Save best model only if Dice Score improves
                if mean_dice > best_dice:
                    best_dice = mean_dice
                    best_model_checkpoint = {
                        'epoch': epoch + 1,
                        'num_epochs': total_epochs,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_dice': best_dice,
                        'train_losses': train_losses,
                        'dice_scores': dice_scores
                    }
                    torch.save(best_model_checkpoint, "model_trained.pth")
                    print(f"New best Dice Score: {best_dice:.4f} (best model saved)")
                else:
                    print(f"No improvement in Dice Score. Best Dice remains: {best_dice:.4f}")

        print("Training complete! 🚀")
        plot_training_results(train_losses, dice_scores)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C). Saving current state...")
        checkpoint = {
            'epoch': epoch + 1,
            'num_epochs': total_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_dice': best_dice,
            'train_losses': train_losses,
            'dice_scores': dice_scores
        }
        torch.save(checkpoint, "checkpoint.pth")
        plot_training_results(train_losses, dice_scores)
        print("Interrupted checkpoint saved as 'checkpoint.pth'. Plots saved.")

def plot_training_results(train_losses, dice_scores, save_path="training_results.png"):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, dice_scores, 'r-', label='Dice Score')
    plt.title('Validation Dice Score')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Training plots saved to {save_path}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()