import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from model import FastDVDnet
from functions_valery import linear_transform   

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Run inference on a sequence")

parser.add_argument("--sequence_path", type=str, required=True, help="Folder containing the frames of the sequence")
parser.add_argument("--output_path", type=str, default="./results/output", help="Where to save the output results")
parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.pth)")

args = parser.parse_args()

# -----------------------
# CONFIGURACIÓN
# -----------------------

# checkpoint = "./results/run_26-03-05_19_20_15/epoch_50.pth"
# sequence_path = "/mnt/bdisk/dewil/loreal_POC2/sequences_for_self-supervised_tests/easier/017"   # carpeta con los 10 frames
# output_path = "./results/output_17"
os.makedirs(args.output_path, exist_ok=True)
preprocessing_parameters_path = os.path.join(args.sequence_path, "pre-processing.txt")

# -----------------------
# CARGAR MODELO
# -----------------------

model = FastDVDnet(num_input_frames=5).to(device)

ckpt = torch.load(args.ckpt, map_location=device)
model.load_state_dict(ckpt["state_dict"])

model.eval()

print("Checkpoint cargado:", args.ckpt)

# -----------------------
# CARGAR FRAMES
# -----------------------

frames = sorted(os.listdir(args.sequence_path))

imgs = []

for f in frames:
    if not f.lower().endswith(("png","tif","jpg")):
        continue

    img = Image.open(os.path.join(args.sequence_path, f)).convert("L")
    img = np.array(img).astype(np.float32) / 255.0
    H = img.shape[0]
    W = img.shape[1]
    H = (H // 8) * 8
    W = (W // 8) * 8
    img = img[:H, :W]
    imgs.append(img)

imgs = np.stack(imgs)

print("Frames cargados:", imgs.shape)

# -----------------------
# INFERENCIA
# -----------------------

recons = []
psnr_vals = []
ssim_vals = []

with torch.no_grad():

    for i in range(len(imgs) - 4):

        window = imgs[i:i+5]
        gt = imgs[i+2]

        x = torch.from_numpy(window).unsqueeze(0).to(device)

        out = model(x)
        out = out.squeeze().cpu().numpy()
        print('Media de out: ', np.mean(out**2))
        recons.append(out)

        psnr_vals.append(psnr(gt, out, data_range=1))
        ssim_vals.append(ssim(gt, out, data_range=1))

        # Save image
        [a,b] = np.loadtxt(preprocessing_parameters_path)
        out_antitransformed = linear_transform(out*9000, a, b, inverse=True)
        #out_img = (out*255).astype(np.uint8)
        print(f"Minimum value of the antitransformed output: {np.min(out_antitransformed)}")
        print(f"Maximum value of the antitransformed output: {np.max(out_antitransformed)}")
        out_int16 = (out_antitransformed).astype(np.int16)
        Image.fromarray(out_int16).save(os.path.join(args.output_path, f"recon_{i:03d}.png"))

# -----------------------
# RESULTS
# -----------------------

print("PSNR medio:", np.mean(psnr_vals))
print("SSIM medio:", np.mean(ssim_vals))

# -----------------------
# VISUALIZACIÓN
# -----------------------

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Input (frame central)")
plt.imshow(imgs[2], cmap="gray")
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Reconstrucción")
plt.imshow(recons[0], cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Error")
plt.imshow(np.abs(imgs[2]-recons[0]), cmap="inferno")
plt.axis("off")

plt.show()