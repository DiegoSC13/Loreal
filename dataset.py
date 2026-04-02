from torch.utils.data import Dataset
import tifffile
import torch
import numpy as np

from pathlib import Path
from torchvision.io import read_image
from utils import *
import imageio.v3 as iio

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

def get_valid_sequences(sequence_paths, out_file="sequences_left_out.txt"):
    """
    Filtra las secuencias que no cumplen con los requisitos:
    - Debe existir pre-processing.txt
    - El valor 'a' debe estar cerca de 1 (abs(a-1) <= 0.2)
    - Debe haber archivos .tif
    - Al menos uno de los canales (o la secuencia completa) debe tener >= 5 frames
    """
    valid_sequences = []
    with open(out_file, "w") as f_out:
        for seq in sequence_paths:
            seq = Path(seq)
            if not seq.is_dir():
                continue
            
            preproc_file = seq / "pre-processing.txt"
            if not preproc_file.exists():
                f_out.write(f"{seq.name}, no pre-processing.txt file\n")
                continue
            
            try:
                a, b = np.loadtxt(preproc_file)
            except Exception as e:
                f_out.write(f"{seq.name}, error reading pre-processing.txt: {e}\n")
                continue

            if np.abs(a-1) > 0.2:
                f_out.write(f"{seq.name}, a={a}\n")
                continue
            
            tif_files = sorted(seq.glob("*.tif"))
            if not tif_files:
                continue
            
            # Verificar si hay suficientes frames en al menos un canal
            names = [f.name for f in tif_files]
            channels = []
            if any("_c0_" in n for n in names):
                channels.append("_c0_")
            if any("_c1_" in n for n in names):
                channels.append("_c1_")
            if not channels:
                channels = [""]
            
            has_enough_frames = False
            for ch in channels:
                if ch:
                    frames = [f for f in tif_files if ch in f.name]
                else:
                    frames = tif_files
                if len(frames) >= 5:
                    has_enough_frames = True
                    break
            
            if has_enough_frames:
                valid_sequences.append((str(seq), float(a), float(b)))
            else:
                f_out.write(f"{seq.name}, not enough frames (min 5)\n")
        valid_sequences = sorted(valid_sequences)
    return valid_sequences

class FastDVDnetDataset(Dataset):
    def __init__(self, sequence_info, patch_size=None, transform=None, data_scale=9000.0):
        """
        Dataset para FastDVDnet:
        - Soporta imágenes .tif
        - Aplica transformación lineal (a,b) desde pre_processing.txt
        - Genera stacks de 5 frames
        - Crop aleatorio opcional
        
        sequence_info: lista de tuples (path, a, b)
        """
        self.patch_size = patch_size
        self.transform = transform
        self.data_scale = data_scale
        self.stacks = []

        for seq_path, a, b in sequence_info:
            seq = Path(seq_path)
            tif_files = sorted(seq.glob("*.tif"))
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

    def make_divisible_by_4(self, img):
        H, W = img.shape[-2], img.shape[-1]
        H4 = (H // 4) * 4
        W4 = (W // 4) * 4
        return img[..., :H4, :W4]

    def _read_tif(self, path):
        """Lee un .tif y lo convierte a tensor [1,H,W] sin normalizar (crudo)"""
        img = iio.imread(str(path))
        img = torch.from_numpy(img).float()

        if img.ndim == 2:
            img = img.unsqueeze(0)
        elif img.ndim == 3:
            img = img.permute(2, 0, 1)
            img = img.mean(dim=0, keepdim=True)

        return img

    def __getitem__(self, idx):
        stack_paths, a, b = self.stacks[idx]

        frames = [self._read_tif(p) for p in stack_paths]
        stack = torch.cat(frames, dim=0)  # [5,H,W]
        stack = self.make_divisible_by_4(stack)
        
        # Valery's linear transform / data_scale
        stack = linear_transform(stack, a, b) / self.data_scale
        stack = torch.clamp(stack, min=0.0)

        # Random crop
        if self.patch_size is not None:
            H, W = stack.shape[1:]
            ph, pw = self.patch_size
            if H >= ph and W >= pw:
                top = torch.randint(0, H - ph + 1, (1,)).item()
                left = torch.randint(0, W - pw + 1, (1,)).item()
                stack = stack[:, top:top + ph, left:left + pw]
        if self.transform:
            stack = self.transform(stack)
        
        target = stack[2:3, :, :].clone()  # Central frame [1,H,W]

        return stack, target  # [5,H,W], [1,H,W]
