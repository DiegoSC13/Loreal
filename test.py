import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from model import FastDVDnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# CONFIGURACIÓN
# -----------------------

checkpoint = "./first_try/ckpts/epoch_91.pth"
sequence_path = "/mnt/bdisk/dewil/loreal_POC2/sequences_for_self-supervised_tests/easier/017"   # carpeta con los 10 frames
output_path = "./first_try/output_17"

os.makedirs(output_path, exist_ok=True)

# -----------------------
# CARGAR MODELO
# -----------------------

model = FastDVDnet(num_input_frames=5).to(device)

ckpt = torch.load(checkpoint, map_location=device)
model.load_state_dict(ckpt["state_dict"])

model.eval()

print("Checkpoint cargado:", checkpoint)

# -----------------------
# CARGAR FRAMES
# -----------------------

frames = sorted(os.listdir(sequence_path))

imgs = []

for f in frames:
    if not f.lower().endswith(("png","tif","jpg")):
        continue

    img = Image.open(os.path.join(sequence_path, f)).convert("L")
    img = np.array(img).astype(np.float32) / 255.0
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

        # guardar imagen
        out_img = (out*255).astype(np.uint8)
        Image.fromarray(out_img).save(
            os.path.join(output_path, f"recon_{i:03d}.png")
        )

# -----------------------
# RESULTADOS
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