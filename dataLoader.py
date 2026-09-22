import os
import glob
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class paired_image_Dataset(Dataset):
    """
    LOL Dataset supporting geometric augmentations (applied jointly to both low/high images)
    and photometric noise augmentations (applied strictly to low-light inputs).
    """
    def __init__(self, low_dir, normal_dir, split='train', crop_size=(256, 256), augment=True):
        super().__init__()
        self.low_dir = low_dir
        self.normal_dir = normal_dir
        self.split = split.lower()
        self.crop_size = crop_size
        self.augment = augment and (self.split == 'train')

        # Color Jitter for exposure/saturation noise on low-light inputs
        self.color_jitter = T.ColorJitter(
            brightness=0.2, 
            contrast=0.2, 
            saturation=0.1, 
            hue=0.05
        )

        self.low_paths, self.high_paths = self._get_image_paths()

    def _get_image_paths(self):
        low_paths, high_paths = [], []

        exts = ('*.png', '*.jpg', '*.jpeg', '*.BMP')
        for ext in exts:
            low_paths.extend(glob.glob(os.path.join(self.low_dir, ext)))

        low_paths = sorted(low_paths)
        for lp in low_paths:
            filename = os.path.basename(lp)
            high_paths.append(os.path.join(self.normal_dir, filename))

        return low_paths, high_paths

    def __len__(self):
        return len(self.low_paths)

    def _apply_joint_augmentations(self, low_img, high_img):
        # 1. Random Crop
        if self.crop_size is not None:
            i, j, h, w = T.RandomCrop.get_params(low_img, output_size=self.crop_size)
            low_img = TF.crop(low_img, i, j, h, w)
            high_img = TF.crop(high_img, i, j, h, w)

        # 2. Random Horizontal Flip
        if random.random() > 0.5:
            low_img = TF.hflip(low_img)
            high_img = TF.hflip(high_img)

        # 3. Random Vertical Flip
        if random.random() > 0.5:
            low_img = TF.vflip(low_img)
            high_img = TF.vflip(high_img)

        # 4. Discrete 90-degree Rotations (0, 90, 180, or 270 degrees)
        rot_deg = random.choice([0, 90, 180, 270])
        if rot_deg > 0:
            low_img = TF.rotate(low_img, rot_deg)
            high_img = TF.rotate(high_img, rot_deg)

        return low_img, high_img

    def __getitem__(self, index):
        low_img = Image.open(self.low_paths[index]).convert('RGB')
        high_img = Image.open(self.high_paths[index]).convert('RGB')

        if self.augment:
            # Apply geometric transforms to both images identically
            low_img, high_img = self._apply_joint_augmentations(low_img, high_img)

            # Apply photometric jitter ONLY to the low-light input (simulates exposure variance)
            # if random.random() > 0.5:
            #     low_img = self.color_jitter(low_img)

        # Convert PIL images to tensors in range [0.0, 1.0]
        low_tensor = TF.to_tensor(low_img)
        high_tensor = TF.to_tensor(high_img)

        return {
            'low': low_tensor,
            'high': high_tensor,
            'name': os.path.basename(self.low_paths[index])
        }


class PairedMixupCollate:
    """
    Custom collate_fn for PyTorch DataLoader that performs paired image Mixup.
    Blends pairs of (low, high) images with the same mixing ratio lambda.
    """
    def __init__(self, alpha=0.2, mixup_prob=0.5):
        self.alpha = alpha
        self.mixup_prob = mixup_prob

    def __call__(self, batch):
        low_list = [item['low'] for item in batch]
        high_list = [item['high'] for item in batch]
        names = [item['name'] for item in batch]

        low_batch = torch.stack(low_list, dim=0)
        high_batch = torch.stack(high_list, dim=0)

        # Apply Mixup conditionally
        if random.random() < self.mixup_prob and low_batch.size(0) > 1:
            # Sample mixing ratio from Beta distribution
            lam = float(torch.distributions.Beta(self.alpha, self.alpha).sample())

            # Generate random permutation indices
            perm = torch.randperm(low_batch.size(0))

            # Blend low-light inputs and target normal-light images identically
            low_batch = lam * low_batch + (1 - lam) * low_batch[perm]
            high_batch = lam * high_batch + (1 - lam) * high_batch[perm]

        return {
            'low': low_batch,
            'high': high_batch,
            'names': names
        }

def get_dataloaders(low_dir_train, normal_dir_train, low_dir_test, normal_dir_test, batch_size=8, crop_size=(256, 256), num_workers=4, use_mixup=False):
    """
    Helper function to instantiate both Train and Test DataLoaders.
    """
    # Training dataset and dataloader
    train_dataset = paired_image_Dataset(
        low_dir=low_dir_train,
        normal_dir=normal_dir_train,
        split='train',
        crop_size=crop_size
    )

    if use_mixup:
        mixup_collate = PairedMixupCollate(alpha=0.2, mixup_prob=0.5)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=mixup_collate,
            pin_memory=True
        )

    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

    # Testing dataset and dataloader (No cropping or shuffling)
    test_dataset = paired_image_Dataset(
        low_dir=low_dir_test,
        normal_dir=normal_dir_test,
        split='test',
        crop_size=None
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader



if __name__ == '__main__':
    low_dir_train = '/home/htoo/Retinexformer/Datasets/LOLv1/Train/input/'
    normal_dir_train = '/home/htoo/Retinexformer/Datasets/LOLv1/Train/target/'
    low_dir_test = '/home/htoo/Retinexformer/Datasets/LOLv1/Test/input/'
    normal_dir_test = '/home/htoo/Retinexformer/Datasets/LOLv1/Test/target/'

    train_loader, test_loader = get_dataloaders(low_dir_train, normal_dir_train, low_dir_test, normal_dir_test)

    for batch in train_loader:
        print("Augmented Train Batch:")
        print("  Low-light input shape :", batch['low'].shape)   # [8, 3, 256, 256]
        print("  Normal-light target shape:", batch['high'].shape) # [8, 3, 256, 256]
        print("  Value ranges - Low min/max :", batch['low'].min().item(), batch['low'].max().item())
        print("  Value ranges - High min/max :", batch['high'].min().item(), batch['high'].max().item())
        break

