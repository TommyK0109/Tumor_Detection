import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
from models.UNetResNet import UNetResNet3D
import nibabel as nib
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
class ValDataset(Dataset):
    def __init__(self, image_dir, original_files):
        self.image_files = sorted(glob.glob(os.path.join(image_dir, "*.npy")))
        self.original_files = original_files
        assert len(self.image_files) == len(self.original_files), "Mismatch between preprocessed and original files"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = np.load(self.image_files[idx])
        if image.shape != (128, 128, 128, 4):
            raise ValueError(f"Unexpected image shape at index {idx}: {image.shape}")
        image = torch.tensor(image, dtype=torch.float32).permute(3, 0, 1, 2)
        return image, idx

def resample_to_original(pred, target_shape=(240, 240, 155)):
    factors = (target_shape[0] / pred.shape[0], target_shape[1] / pred.shape[1], target_shape[2] / pred.shape[2])
    return zoom(pred, factors, order=0).astype(np.uint8)

def plot_prediction(image, pred, slice_idx=64, save_path=None):
    """
    Plot a slice of the predicted segmentation mask overlaid on the input image.
    - image: (128, 128, 128, 4) numpy array (T1, T2, FLAIR, T1ce)
    - pred: (128, 128, 128) predicted mask
    - slice_idx: Slice to visualize (default: middle slice)
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Select T1ce modality (index 3) for visualization
    t1ce_slice = image[:, :, slice_idx, 3]
    pred_slice = pred[:, :, slice_idx]

    # Plot T1ce slice
    ax[0].imshow(t1ce_slice, cmap="gray")
    ax[0].set_title("T1ce Image")
    ax[0].axis("off")

    # Plot predicted mask
    ax[1].imshow(pred_slice, cmap="jet", vmin=0, vmax=3)
    ax[1].set_title("Predicted Mask")
    ax[1].axis("off")

    # Overlay prediction on T1ce
    ax[2].imshow(t1ce_slice, cmap="gray")
    ax[2].imshow(pred_slice, cmap="jet", alpha=0.5, vmin=0, vmax=3)
    ax[2].set_title("Overlay")
    ax[2].axis("off")

    if save_path:
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
    plt.show()

# Set paths
output_path = r"/data/BraTS2020_Preprocessed_Validate"
val_path = r"/data/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData"

# Get original NIfTI files for affines and patient IDs
val_T1_list = sorted(glob.glob(os.path.join(val_path, "*", "*_t1.nii")))

# Create dataset and loader
val_dataset = ValDataset(os.path.join(output_path, "val/images"), val_T1_list)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

print(f"Number of validation samples: {len(val_dataset)}")

# Load the trained model
model = UNetResNet3D(resnet_type="resnet50_3d", in_channels=4, out_channels=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("best_unet_resnet.pth", map_location=device))
model.eval()
model.to(device)

# Directory for saving predictions and plots
output_dir = os.path.join(output_path, "val/predictions")
plot_dir = os.path.join(output_path, "val/plots")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# Inference loop
visualize_samples = 5  # Number of samples to visualize
visualized = 0

with torch.no_grad():
    for images, idx in val_loader:
        images = images.to(device)
        outputs = model(images)
        pred = torch.argmax(outputs, dim=1).cpu().numpy()[0]  # (128, 128, 128)

        # Resample to original size for saving
        pred_resampled = resample_to_original(pred)

        # Load affine from original NIfTI file
        original_nifti = nib.load(val_T1_list[idx])
        affine = original_nifti.affine

        # Extract patient ID from file name
        patient_id = os.path.basename(val_T1_list[idx]).split("_t1.nii")[0]

        # Save prediction with correct affine and naming
        pred_file = os.path.join(output_dir, f"{patient_id}.nii.gz")
        nib.save(nib.Nifti1Image(pred_resampled, affine), pred_file)
        print(f"Saved prediction: {pred_file}")

        # Visualize a few samples (before resampling)
        if visualized < visualize_samples:
            plot_path = os.path.join(plot_dir, f"{patient_id}_slice.png")
            image_np = images.cpu().numpy()[0].transpose(1, 2, 3, 0)  # (C, D, H, W) -> (D, H, W, C)
            plot_prediction(image_np, pred, slice_idx=64, save_path=plot_path)
            visualized += 1

print("Validation inference complete!")

