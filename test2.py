"""
test.py — Inferencia con FastDVDnet, guarda imágenes denoised como .tif

Recorre base_dir con la misma lógica que FastDVDnetDataset:
  base_dir/
    secuencia_1/
      pre-processing.txt
      frame_001.tif
      frame_002.tif
      ...
    secuencia_2/
      ...

Por cada frame central de cada secuencia, guarda el resultado denoised en:
  output_path/
    secuencia_1/
      frame_003_denoised.tif
      frame_004_denoised.tif
      ...

Uso:
    python test.py \
        --ckpt ./results/train_XX/epoch_200.pth \
        --base_dir /ruta/a/base_dir \
        --output_path ./test_results \
        [--patch_size 256 256]
"""

import argparse
import numpy as np
import torch
import tifffile
import imageio.v3 as iio
from pathlib import Path

from model import FastDVDnet, SureWrapper


# ─────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────

def read_tif(path: Path) -> torch.Tensor:
    """Lee un .tif y devuelve tensor [1, H, W] normalizado en [0, 1]."""
    img = iio.imread(str(path))
    img = torch.from_numpy(img.astype(np.float32))
    if img.ndim == 2:
        img = img.unsqueeze(0)
    elif img.ndim == 3:
        img = img.permute(2, 0, 1).mean(dim=0, keepdim=True)
    img = img / 255.0
    return img


def load_model(ckpt_path: str, device: torch.device) -> SureWrapper:
    """Carga FastDVDnet + SureWrapper desde checkpoint."""
    model = FastDVDnet(num_input_frames=5).to(device)
    wrapper = SureWrapper(model).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)

    epoch_info = f", época {ckpt['epoch']}" if "epoch" in ckpt else ""
    print(f"Checkpoint cargado: {ckpt_path}{epoch_info}")

    wrapper.eval()
    return wrapper


def iter_sequences(base_dir: Path):
    """
    Genera (seq_name, channel, stack_paths) con la misma lógica de filtrado
    que FastDVDnetDataset: requiere pre-processing.txt y |a-1| <= 0.2.
    """
    for seq in sorted(base_dir.iterdir()):
        if not seq.is_dir():
            print(f"  [skip] {seq.name}not seq.is_dir()")
            continue

        preproc_file = seq / "pre-processing.txt"
        if not preproc_file.exists():
            print(f"  [skip] {seq.name}: no pre-processing.txt")
            continue

        a, b = np.loadtxt(preproc_file)
        if np.abs(a - 1) > 0.2:
            print(f"  [skip] {seq.name}: a={a:.3f} fuera de rango")
            continue

        tif_files = sorted(seq.glob("*.tif"))
        if not tif_files:
            print(f"  [skip] {seq.name}: no hay .tif")
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
            frames = sorted(f for f in tif_files if ch in f.name) if ch else tif_files
            if len(frames) < 5:
                print(f"  [skip] {seq.name} canal '{ch or 'single'}': solo {len(frames)} frames")
                continue

            for i in range(2, len(frames) - 2):
                stack_paths = [
                    frames[i - 2],
                    frames[i - 1],
                    frames[i],       # frame central
                    frames[i + 1],
                    frames[i + 2],
                ]
                yield seq.name, ch, stack_paths


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Inferencia FastDVDnet → .tif denoised")
    parser.add_argument("--ckpt",        type=str, required=True,
                        help="Ruta al checkpoint .pth")
    parser.add_argument("--base_dir",    type=str, required=True,
                        help="Directorio raíz con subdirectorios de secuencias")
    parser.add_argument("--output_path", type=str, default="./test_results",
                        help="Directorio donde se guardan los .tif denoised")
    parser.add_argument("--patch_size",  type=int, nargs=2, default=None,
                        help="Crop central para inferencia, ej: --patch_size 256 256")
    parser.add_argument("--test_indexes", type=str, default=None,
                        help="Ruta al test_indices.txt generado durante el entrenamiento")
    args = parser.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path(args.base_dir)
    out_root = Path(args.output_path)
    out_root.mkdir(parents=True, exist_ok=True)
    patch_size = tuple(args.patch_size) if args.patch_size else None

    print(f"Device:   {device}")
    print(f"Base dir: {base_dir}")
    print(f"Salida:   {out_root}\n")

    wrapper = load_model(args.ckpt, device)

    # Instanciar dataset igual que en train (sin patch_size, queremos frames completos)
    dataset = FastDVDnetDataset(base_dirs=[args.base_dir], patch_size=None)
    print(f"Stacks totales en dataset: {len(dataset.stacks)}")
 
    # Filtrar por índices de test si se proporcionan
    if args.test_indices:
        indices = np.loadtxt(args.test_indices, dtype=int)
        stacks_to_process = [dataset.stacks[i] for i in indices]
        print(f"Procesando {len(stacks_to_process)} stacks del conjunto de test\n")
    else:
        stacks_to_process = dataset.stacks
        print(f"Procesando todos los {len(stacks_to_process)} stacks (no se especificó test_indices)\n")
 
    n_saved = 0
 
    with torch.no_grad():
        for stack_paths, a, b in stacks_to_process:
            # Directorio de salida: replica el nombre de la secuencia
            seq_name = stack_paths[2].parent.name  # nombre del subdirectorio del frame central
            seq_out_dir = out_root / seq_name
            seq_out_dir.mkdir(parents=True, exist_ok=True)

            # Leer stack
            frames = [read_tif(p) for p in stack_paths]
            stack  = torch.cat(frames, dim=0)          # [5, H, W]

            # Crop central (reproducible en test, no aleatorio)
            if patch_size is not None:
                H, W = stack.shape[1], stack.shape[2]
                ph, pw = patch_size
                top  = max(0, (H - ph) // 2)
                left = max(0, (W - pw) // 2)
                stack = stack[:, top:top+ph, left:left+pw]

            stack     = stack.unsqueeze(0).to(device)  # [1, 5, H, W]
            y_central = stack[:, 2:3, :, :]            # [1, 1, H, W]

            wrapper.set_context(stack)
            output = wrapper(y_central)                # [1, 1, H, W]

            # Guardar como .tif uint16 en escala original [0, 255]
            denoised_np = output[0, 0].cpu().numpy()
            denoised_uint16 = (denoised_np * 255.0).clip(0, 65535).astype(np.uint16)

            stem = stack_paths[2].stem  # nombre del frame central sin extensión
            out_path = seq_out_dir / f"{stem}_denoised.tif"
            tifffile.imwrite(str(out_path), denoised_uint16)
            n_saved += 1

        print(f"Guardados {n_saved} frames denoised en {out_root}")


if __name__ == "__main__":
    main()
