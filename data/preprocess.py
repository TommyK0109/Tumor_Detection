import os
import glob
import numpy as np
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import zoom
import random
from tqdm import tqdm  # Add progress bar

# Set paths
train_path = r"A:\Testing\data\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
output_path = r"A:\Testing\data\BraTS2020_Preprocessed"

os.makedirs(os.path.join(output_path, "images"), exist_ok=True)
os.makedirs(os.path.join(output_path, "masks"), exist_ok=True)

def get_file_lists(dataset_path):
    t1_list = sorted(glob.glob(os.path.join(dataset_path, "*", "*_t1.nii")))
    t2_list = sorted(glob.glob(os.path.join(dataset_path, "*", "*_t2.nii")))
    flair_list = sorted(glob.glob(os.path.join(dataset_path, "*", "*_flair.nii")))
    t1ce_list = sorted(glob.glob(os.path.join(dataset_path, "*", "*_t1ce.nii")))
    mask_list = sorted(glob.glob(os.path.join(dataset_path, "*", "*_seg.nii")))

    lengths = [len(t1_list), len(t2_list), len(flair_list), len(t1ce_list), len(mask_list)]
    if len(set(lengths)) != 1:
        raise ValueError(f"Mismatched file counts: T1={lengths[0]}, T2={lengths[1]}, "
                         f"FLAIR={lengths[2]}, T1ce={lengths[3]}, Mask={lengths[4]}")

    return t1_list, t2_list, flair_list, t1ce_list, mask_list

def calculate_normalization_stats(dataset_path, sample_count=10):
    t1_list, t2_list, flair_list, t1ce_list, _ = get_file_lists(dataset_path)
    modality_stats = {}

    # Calculate stats for each modality
    for modality_name, modality_list in [
        ('t1', t1_list), ('t2', t2_list),
        ('flair', flair_list), ('t1ce', t1ce_list)
    ]:
        print(f"Calculating statistics for {modality_name}...")
        samples = []
        # Use a subset to save memory
        for idx in range(min(sample_count, len(modality_list))):
            try:
                img = nib.load(modality_list[idx]).get_fdata().astype(np.float32)
                # Skip zeros (background) for better normalization
                non_zero_values = img[img > 0].reshape(-1)
                if len(non_zero_values) > 0:
                    samples.append(non_zero_values)
            except Exception as e:
                print(f"Error loading {modality_list[idx]}: {e}")
                continue

        if samples:
            all_samples = np.concatenate(samples)
            # Use percentiles to avoid extreme outliers
            modality_stats[modality_name] = {
                'min': np.percentile(all_samples, 1),
                'max': np.percentile(all_samples, 99)
            }
            print(f"{modality_name} stats: min={modality_stats[modality_name]['min']:.2f}, "
                  f"max={modality_stats[modality_name]['max']:.2f}")
        else:
            print(f"Warning: No valid samples found for {modality_name}")
            modality_stats[modality_name] = {'min': 0, 'max': 1}

    return modality_stats


# Function to normalize a single volume
def normalize_volume(volume, modality, stats, method='minmax'):
    if method == 'minmax':
        # Min-max scaling to [0,1]
        min_val = stats[modality]['min']
        max_val = stats[modality]['max']
        normalized = (volume - min_val) / (max_val - min_val)
        return np.clip(normalized, 0, 1)
    elif method == 'zscore':
        # Z-score normalization (mean=0, std=1)
        mean_val = stats[modality]['mean']
        std_val = stats[modality]['std']
        return (volume - mean_val) / std_val
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# Function to dynamically center crop a volume
def center_crop(volume, target_shape=(128, 128, 128)):
    # Get current shape
    current_shape = volume.shape

    # Calculate start indices for cropping
    start_indices = [(current_shape[i] - target_shape[i]) // 2 for i in range(3)]

    # Perform cropping
    cropped = volume[
              start_indices[0]:start_indices[0] + target_shape[0],
              start_indices[1]:start_indices[1] + target_shape[1],
              start_indices[2]:start_indices[2] + target_shape[2]
              ]

    return cropped


# Function to resample a volume to target shape
def resample(volume, target_shape=(128, 128, 128)):
    factors = (target_shape[0] / volume.shape[0],
               target_shape[1] / volume.shape[1],
               target_shape[2] / volume.shape[2])
    return zoom(volume, factors, order=1).astype(np.float32)


# Simple data augmentation functions
def random_flip(volume, mask, axis=0, p=0.5):
    if random.random() < p:
        return np.flip(volume, axis=axis), np.flip(mask, axis=axis)
    return volume, mask


def random_rotate(volume, mask, angle_range=(-10, 10), axis=(0, 1), p=0.5):
    if random.random() < p:
        pass
    return volume, mask


# Function to process the training dataset
def process_dataset(dataset_path, output_path, modality_stats, augment=False):
    print("Processing training dataset...")

    # Get file lists
    t1_list, t2_list, flair_list, t1ce_list, mask_list = get_file_lists(dataset_path)

    # Process each sample
    for idx in tqdm(range(len(t1_list)), desc="Processing"):
        try:
            # Extract sample ID for better file naming
            sample_id = os.path.basename(os.path.dirname(t1_list[idx]))

            # Load images
            t1 = nib.load(t1_list[idx]).get_fdata().astype(np.float32)
            t2 = nib.load(t2_list[idx]).get_fdata().astype(np.float32)
            flair = nib.load(flair_list[idx]).get_fdata().astype(np.float32)
            t1ce = nib.load(t1ce_list[idx]).get_fdata().astype(np.float32)
            mask = nib.load(mask_list[idx]).get_fdata().astype(np.uint8)

            # Ensure all volumes have the same shape
            original_shape = t1.shape
            if not all(img.shape == original_shape for img in [t2, flair, t1ce, mask]):
                print(f"Warning: Inconsistent shapes in {sample_id}. Skipping...")
                continue

            # Choose between cropping and resampling based on original size
            if all(dim >= 128 for dim in original_shape):
                # Center crop
                t1 = center_crop(t1)
                t2 = center_crop(t2)
                flair = center_crop(flair)
                t1ce = center_crop(t1ce)
                mask = center_crop(mask)
            else:
                # Resample
                t1 = resample(t1)
                t2 = resample(t2)
                flair = resample(flair)
                t1ce = resample(t1ce)
                mask = resample(mask).astype(np.uint8)

            # Normalize each modality using its own statistics
            t1 = normalize_volume(t1, 't1', modality_stats)
            t2 = normalize_volume(t2, 't2', modality_stats)
            flair = normalize_volume(flair, 'flair', modality_stats)
            t1ce = normalize_volume(t1ce, 't1ce', modality_stats)

            mask[mask == 4] = 3

            # Verify mask values
            unique_vals = np.unique(mask)
            if not all(val in [0, 1, 2, 3] for val in unique_vals):
                print(f"Warning: Invalid mask values in {sample_id}: {unique_vals}")
                # Correct any invalid values
                mask[~np.isin(mask, [0, 1, 2, 3])] = 0

            # Apply augmentations if requested
            if augment:
                # Apply random flips
                t1, mask = random_flip(t1, mask, axis=0)
                t2, mask = random_flip(t2, mask, axis=0)
                flair, mask = random_flip(flair, mask, axis=0)
                t1ce, mask = random_flip(t1ce, mask, axis=0)

                # Additional augmentations could be added here

            # Combine all 4 modalities
            combined_x = np.stack([t1, t2, flair, t1ce], axis=3)  # Shape: (128, 128, 128, 4)

            # Save processed images and masks
            np.save(os.path.join(output_path, f"images/{sample_id}.npy"), combined_x)
            np.save(os.path.join(output_path, f"masks/{sample_id}.npy"), mask)

            # Free memory
            del t1, t2, flair, t1ce, mask, combined_x

        except Exception as e:
            print(f"Error processing {t1_list[idx]}: {e}")
            continue


# Main execution
def main():
    print("Starting BraTS2020 preprocessing...")
    modality_stats = calculate_normalization_stats(train_path, sample_count=10)
    process_dataset(train_path, output_path, modality_stats, augment=True)

    print("Preprocessing complete!")


if __name__ == "__main__":
    main()