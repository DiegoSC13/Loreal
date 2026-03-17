from torch.utils.data import Dataset
import tifffile
import torch
import numpy as np

from pathlib import Path
from torchvision.io import read_image
from functions_valery import *
from utils import *
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

        #Just to see which sequences I'm not using
        out_file = "sequences_left_out.txt"
        with open(out_file, "w") as f_out:
            for base_dir in base_dirs:
                base_dir = Path(base_dir)
                # print(f'{base_dir=}')
                if not base_dir.exists():
                    # print(f'base_dirs does not exist, skipping')
                    continue
                for seq in base_dir.iterdir():
                    if not seq.is_dir():
                        # print(f'Sequence {seq} is not a directory, skipping')
                        continue
                    preproc_file = seq / "pre-processing.txt"
                    # print("Chequeando archivo:", preproc_file.resolve())
                    if not preproc_file.exists():
                        print(f'There is not pre-processing.txt file, skipping')
                        f_out.write(f"{seq.name}, no pre-processing.txt file\n")
                        continue
                    a, b = np.loadtxt(preproc_file)
                    if np.abs(a-1) > 0.2:
                        print(f'In sequence {seq} the value of a is too far from 1, skipping')
                        f_out.write(f"{seq.name}, a={a}\n")
                        continue
                    tif_files = sorted(seq.glob("*.tif"))
                    if not tif_files:
                        print(f"No tif files found in {seq}, skipping")
                        continue
                    names = [f.name for f in tif_files]
                    channels = []
                    if any("_c0_" in n for n in names):
                        channels.append("_c0_")
                    if any("_c1_" in n for n in names):
                        channels.append("_c1_")
                    if not channels:
                        channels = [""]
                    for ch in channels:
                        if ch:
                            frames = sorted(f for f in tif_files if ch in f.name)
                        else:
                            frames = sorted(tif_files)
                        if len(frames) < 5:
                            print(f'The number of sequence frames in {seq} for channel {ch or "single"} is {len(frames)}, which is lower than 5. Skipping')
                            f_out.write(f"{seq.name}, Channel={ch or 'single'}, Number of frames={len(frames)}\n")
                            continue

                        # ahora sí armamos stacks solo dentro del canal
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

        img = img / 255.0  

        return img

    def __getitem__(self, idx):
        stack_paths, a, b = self.stacks[idx]

        frames = [self._read_tif(p) for p in stack_paths]
        stack = torch.cat(frames, dim=0)  # [5,H,W]

        # Valery's linear transform
        # stack = linear_transform(stack, a, b) # / 9000

        # Resampling, this way all sequences end up on the same noise distribution (if they have the same x (not the case here))
        # stack = resample_poisson_sequence(stack, a) #Is 1.4 by default 

        # Random crop
        if self.patch_size is not None:
            H, W = stack.shape[1:]
            ph, pw = self.patch_size

            if H >= ph and W >= pw:
                top = torch.randint(0, H - ph + 1, (1,)).item()
                left = torch.randint(0, W - pw + 1, (1,)).item()
                stack = stack[:, top:top + ph, left:left + pw]

        target = stack[2:3, :, :].clone()  # Central frame [1,H,W]

       # print("Input min/max:", stack.min().item(), stack.max().item())
       # print("Target min/max:", target.min().item(), target.max().item())
        
        return stack, target  # [5,H,W], [1,H,W]
