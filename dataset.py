from torch.utils.data import Dataset
import tifffile
import torch
import numpy as np

from pathlib import Path
from torchvision.io import read_image
from functions_valery import *
import imageio.v3 as iio

#No hace nada, simplemente devuelve imagen ruidosa.

class LorealDataset(Dataset):
    def __init__(self, image_paths, transform=None, patch_size=None):
        self.image_paths = image_paths
        self.transform = transform
        self.patch_size = patch_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = tifffile.imread(self.image_paths[idx]).astype(np.float32)

        img = torch.from_numpy(img)

        if self.patch_size:
            img = self.random_crop(img)

        if self.transform:
            img = self.transform(img)

        return img

    def random_crop(self, img):
        H, W = img.shape
        ph, pw = self.patch_size
        i = torch.randint(0, H - ph, (1,)).item()
        j = torch.randint(0, W - pw, (1,)).item()
        return img[i:i+ph, j:j+pw]

class FastDVDnetDataset(Dataset):
    def __init__(self, base_dirs, patch_size=None):
        """
        Dataset para FastDVDnet:
        - Soporta imágenes .tif
        - Aplica transformación lineal (a,b) desde pre_processing.txt
        - Genera stacks de 5 frames
        - Crop aleatorio opcional
        """

        self.patch_size = patch_size
        self.stacks = []

        for base_dir in base_dirs:
            base_dir = Path(base_dir)
            if not base_dir.exists():
                continue

            for seq in base_dir.iterdir():
                if not seq.is_dir():
                    continue

                preproc_file = seq / "pre-processing.txt"
                if not preproc_file.exists():
                    continue

                a, b = np.loadtxt(preproc_file)

                for channel_prefix in ["image_c0_", "image_c1_"]:
                    frames = sorted(seq.glob(f"{channel_prefix}*.tif"))

                    if len(frames) < 5:
                        continue

                    for i in range(2, len(frames) - 2):
                        stack_paths = [
                            frames[i - 2],
                            frames[i - 1],
                            frames[i],
                            frames[i + 1],
                            frames[i + 2],
                        ]

                        self.stacks.append((stack_paths, float(a), float(b)))

    def __len__(self):
        return len(self.stacks)

    def _read_tif(self, path):
        """Lee un .tif y lo convierte a tensor [1,H,W] normalizado"""
        img = iio.imread(str(path))
        img = torch.from_numpy(img).float()

        if img.ndim == 2:
            img = img.unsqueeze(0)
        elif img.ndim == 3:
            img = img.permute(2, 0, 1)
            img = img.mean(dim=0, keepdim=True)

        if img.max() > 1:
            img = img / 65535.0  # asumir uint16

        return img

    def __getitem__(self, idx):
        stack_paths, a, b = self.stacks[idx]

        frames = [self._read_tif(p) for p in stack_paths]
        stack = torch.cat(frames, dim=0)  # [5,H,W]

        # Transformación lineal
        stack = linear_transform(stack, a, b)

        # Crop aleatorio
        if self.patch_size is not None:
            H, W = stack.shape[1:]
            ph, pw = self.patch_size

            if H >= ph and W >= pw:
                top = torch.randint(0, H - ph + 1, (1,)).item()
                left = torch.randint(0, W - pw + 1, (1,)).item()
                stack = stack[:, top:top + ph, left:left + pw]

        target = stack[2:3, :, :].clone()  # frame central [1,H,W]
        
        return stack, target  # [5,H,W], [1,H,W]
