
from argparse import ArgumentParser
import os
from datetime import datetime
from matplotlib import pyplot as plt
import logging
import sys
import shlex
import platform
import time
import numpy as np
import random

import torch
from torch.utils.data import DataLoader, random_split

from dataset import LorealDataset, FastDVDnetDataset, get_valid_sequences
from new_model import FastDVDnet, SureWrapper
from losses import get_loss
from physics import get_physics
from deepinv.loss.r2r import R2RLoss, R2RModel
from utils import *

# This allows for accelerated Tensor Core training; cryoDRGN uses it. I've never been able to use it, but I want to try it someday.
# try:
#     import apex.amp as amp  
# except ImportError:
#     pass

def check_checkpoint_loading_with_magnitude(model, ckpt):
    """
    Para cada parámetro del modelo:
      - Verifica si existe en el checkpoint
      - Compara shape
      - Imprime magnitud media absoluta (mean(abs(param)))
    """
    for name, param in model.named_parameters():
        if name in ckpt:
            if param.shape == ckpt[name].shape:
                status = "Cargado correctamente"
                mag = ckpt[name].abs().mean().item()
            else:
                status = f"Shape mismatch (modelo {param.shape} vs ckpt {ckpt[name].shape})"
                mag = ckpt[name].abs().mean().item()
        else:
            status = "✖ No existe en checkpoint, inicializado desde cero"
            mag = param.abs().mean().item()
        
        print(f"{name:50s} | {status:50s} | mean(abs)={mag:.6f}")

parser = ArgumentParser()
parser.add_argument("--image_paths", type=str, nargs='+', required=True,
                    help="Lista de rutas a las imágenes .tif")
parser.add_argument("--output_path", type=str, required=True, help="Directorio donde se crea carpeta de checkpoints")
parser.add_argument("--ckpt", type=str, default=None, help="Ruta a checkpoint preentrenado") # Pretrained model path
parser.add_argument("--loss", type=str, choices=("sure", "pure", "pgure", "unsure", "unpgure", "r2r_g", "r2r_p"), required=True) # "noise2score",# "unsure")
parser.add_argument("--sigma", type=float, default=None) #Gaussian std
parser.add_argument("--gamma", type=float, default=None) #Poisson scalar factor
parser.add_argument("--alpha", type=float, default=0.15) #R2R recorruption factor
# parser.add_argument("--mc_iter", type=int, default=1) #Era para el estimador de Monte Carlo de la divergencia, pero no veo que lo usen en las losses SURE-based
parser.add_argument("--step_size", type=float, default=(1e-5, 1e-5), help="Gradient step size") #UNSURE and PG-UNSURE 
parser.add_argument("--momentum", type=float, default=(0.9, 0.9), help="Gradient momentum")    #UNSURE and PG-UNSURE 

parser.add_argument("--batch_size", type=int, default=32) #Default of SSIBench is 32, not sure if it's the best option here
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--scheduler_step_size", type=int, default=50)
parser.add_argument("--scheduler_gamma", type=float, default=0.1)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--patch_size", type=int, nargs=2, default=None,
                    help="Tamaño del recorte aleatorio, ej: --patch_size 128 128")
parser.add_argument("--transform", type=str, default=None,
                    help="Nombre de la transformación a aplicar (opcional)")
parser.add_argument("--loss_scaler", type=float, default=1.0, help="Factor para escalar la pérdida antes del backward (para evitar vanishing gradients en escalas pequeñas)")
parser.add_argument("--data_scale", type=float, default=9000.0, help="Factor divisor para los datos tras la transformacion lineal (a,b)")
args = parser.parse_args()

#Creo directorio con fechas para no sobreescribir
timestamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
save_path = os.path.join(args.output_path, f"train_{timestamp}")
os.makedirs(save_path, exist_ok=True)

log_path = os.path.join(args.output_path, f"log_{timestamp}.log")

logger = logging.getLogger(__name__)
#logger.info(f"Python version: {platform.python_version()}")
print(f"Python version: {platform.python_version()}")
print(f"Command used: {shlex.join(sys.argv)}")

# Set the global random seed and select device
# torch.manual_seed(int(time.time())) # Para tener distinta semilla en cada entrenamiento
generator = torch.Generator().manual_seed(42) # Para tener reproducibilidad
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Checkpoints dir
ckpt_path = os.path.join(save_path, "ckpts") 
os.makedirs(ckpt_path, exist_ok=True)

# Memory/efficiency limitations
batch_size = args.batch_size if torch.cuda.is_available() else 1

# Dataset preparation
patch_size = tuple(args.patch_size) if args.patch_size else None

root_dir = '/mnt/bdisk/dewil/loreal_POC2/sequences_almost_Poisson'
all_sequences_paths = []
for root, dirs, files in os.walk(root_dir, followlinks=True):
    # excluir check del recorrido
    dirs[:] = [d for d in dirs if d != "check"]
    for d in dirs:
        seq_path = os.path.join(root, d)
        all_sequences_paths.append(seq_path)

# Filtrar secuencias válidas antes de dividir
valid_sequences_info = get_valid_sequences(all_sequences_paths, out_file=os.path.join(save_path, "sequences_left_out.txt"))

# Shuffle and split sequences
random.seed(42)
random.shuffle(valid_sequences_info)

n_total = len(valid_sequences_info)
n_train = int(0.7 * n_total)
n_val = int(0.15 * n_total)
n_test = n_total - n_train - n_val

train_info = valid_sequences_info[:n_train]
val_info = valid_sequences_info[n_train:n_train + n_val]
test_info = valid_sequences_info[n_train + n_val:]

print(f"Total valid sequences: {n_total}")
print(f"Train sequences: {len(train_info)}")
print(f"Val sequences: {len(val_info)}")
print(f"Test sequences: {len(test_info)}")

train_dataset = FastDVDnetDataset(sequence_info=train_info, patch_size=patch_size, data_scale=args.data_scale)
val_dataset = FastDVDnetDataset(sequence_info=val_info, patch_size=patch_size, data_scale=args.data_scale)
test_dataset = FastDVDnetDataset(sequence_info=test_info, patch_size=patch_size, data_scale=args.data_scale)

print(f"Train dataset size (stacks): {len(train_dataset)}")
print(f"Val dataset size (stacks): {len(val_dataset)}")
print(f"Test dataset size (stacks): {len(test_dataset)}")

# Dataloader preparation
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
val_dataloader   = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
test_dataloader  = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

# Guardar secuencias para referencia
with open(os.path.join(save_path, "train_sequences.txt"), "w") as f:
    for s_info in train_info:
        f.write(f"{s_info[0]}\n")

with open(os.path.join(save_path, "val_sequences.txt"), "w") as f:
    for s_info in val_info:
        f.write(f"{s_info[0]}\n")

with open(os.path.join(save_path, "test_sequences.txt"), "w") as f:
    for s_info in test_info:
        f.write(f"{s_info[0]}\n")

for x, target in train_dataloader:
    print("x shape after dataloader:", x.shape)
    print("Target shape after dataloader:", target.shape)
    break  # solo miro el primer batch

# Choose model
model = FastDVDnet(num_input_frames=5).to(device)
# Wrapper que adapta FastDVDnet (5 frames → 1 frame) a la interfaz que espera SurePoissonLoss
wrapper = SureWrapper(model).to(device)

# Choose physics
physics = get_physics(
    loss_name=args.loss,
    sigma=args.sigma,
    gamma=args.gamma,
    device=device
)
# Choose loss 
loss_fn = get_loss(
        loss_name=args.loss,
        device=device,
        sigma=args.sigma,
        gamma=args.gamma, 
        alpha=args.alpha,
        # mc_iter=args.mc_iter
        step_size = args.step_size,  #UNSURE
        momentum  = args.momentum   #UNSURE
    )

# Adapt model if using R2RLoss
if isinstance(loss_fn, R2RLoss):
    wrapper = loss_fn.adapt_model(wrapper)

# choose optimizer and scheduler
optimizer = torch.optim.Adam(wrapper.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma) #Quiero multiplicar el lr por 0.1 cada 50 epocas

if args.ckpt:
    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)  # compatible con ambos formatos
    model.load_state_dict(state_dict, strict=False)
    check_checkpoint_loading_with_magnitude(model, state_dict)
    print("Loaded model weights (optimizer reinitialized).")

epoch_losses = []
val_losses = []
for epoch in range(args.epochs):
    wrapper.train()
    running_loss = 0.0
    # Antes del loop de batches:
    grad_accum = {name: 0.0 for name, p in model.named_parameters() if p.requires_grad}
    n_batches = 0
    for i, (stack, target) in enumerate(train_dataloader):
        stack = stack.to(device)           # [B, 5, H, W]
        y_central = stack[:, 2:3, :, :]   # [B, 1, H, W]

        if isinstance(wrapper, R2RModel):
            wrapper.model.set_context(stack)
        else:
            wrapper.set_context(stack)

        optimizer.zero_grad()

        # R2RLoss requires physics and update_parameters=True during training
        if isinstance(loss_fn, R2RLoss):
            output = wrapper(y_central, physics, update_parameters=True)
        else:
            output = wrapper(y_central)

        loss = loss_fn(y_central, output, physics, wrapper).mean()
        (loss * args.loss_scaler).backward()
        
        # 1. Gradient Clipping: Vital para evitar que SURE dispare los gradientes
        torch.nn.utils.clip_grad_norm_(wrapper.parameters(), max_norm=1.0)
        
        optimizer.step()

        running_loss += loss.item()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_accum[name] += param.grad.abs().mean().item()
        n_batches += 1

    # # Diagnóstico de gradientes (una vez por época, último batch)
    # for name, param in model.named_parameters():
    #     if param.requires_grad and param.grad is not None:
    #         print(f"{name} | grad={param.grad.abs().mean():.2e} | val={param.abs().mean():.2e}")

    # Validation loop
    wrapper.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, (stack, target) in enumerate(val_dataloader):
            stack = stack.to(device)
            y_central = stack[:, 2:3, :, :]
            
            if isinstance(wrapper, R2RModel):
                wrapper.model.set_context(stack)
            else:
                wrapper.set_context(stack)

            if isinstance(loss_fn, R2RLoss):
                wrapper.training = True
                output = wrapper(y_central, physics, update_parameters=True)
                wrapper.training = False
            else:
                output = wrapper(y_central)

            loss = loss_fn(y_central, output, physics, wrapper).mean()
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0
    val_losses.append(avg_val_loss)

    epoch_loss = running_loss / len(train_dataloader)
    epoch_losses.append(epoch_loss)
    scheduler.step()    
    for name, avg_grad in grad_accum.items():
        print(f"{name} | grad_mean_epoch={avg_grad / n_batches:.2e}")
    print(
    f"Epoch {epoch+1}, "
    f"Loss: {epoch_loss:.8f}, Val Loss: {avg_val_loss:.8f}, "
    f"lr: {optimizer.param_groups[0]['lr']:.2e}\n"
    )
    logger.info(f"Epoch {epoch+1}/{args.epochs} - Loss: {epoch_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

    #Printeo cada 10 epochs
    if (epoch+1) % 10 == 0: 
        this_ckpt_path = os.path.join(ckpt_path, f"epoch_{epoch+1}.pth")
        torch.save({
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "loss": running_loss/len(train_dataloader)
        }, this_ckpt_path)

plt.figure()
plt.plot(epoch_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.grid(True)
plt.show()
plt.savefig(os.path.join(save_path, "loss_plot.png"), dpi=200)

####################################################################

# While I'm debugging I shouldn't use the trainer, after I get everything to work it might be a good idea change to this (not sure)
# Initialize DeepInverse trainer
# trainer = dinv.Trainer(
#     model=model,
#     epochs=args.epochs,
#     scheduler=scheduler,
#     physics=physics,
#     losses=loss_fn,
#     optimizer=optimizer,
#     device=device,
#     train_dataloader=train_dataloader,
#     eval_dataloader=test_dataloader,
#     compute_eval_losses=True,  # use self-supervised loss for evaluation
#     early_stop_on_losses=True,  # stop using self-supervised eval loss
#     metrics=None,  # no supervised metrics
#     early_stop=20,  # early stop using the self-supervised loss on the test set
#     plot_images=False, #Error when dim(y) != dim(x), that's the case in FastDVDnet
#     save_path=save_path,
#     verbose=verbose,
#     show_progress_bar=False,  # disable progress bar for better vis in sphinx gallery.
# )

# Train the network
# model = trainer.train()
# print(trainer.model is model)



# model.eval()  # importante
# with torch.no_grad():
#     batch = next(iter(train_dataloader))  # toma el primer batch
#     if isinstance(batch, (tuple, list)):
#         y = batch[0].to(device)  # ajusta según tu dataset
#     else:
#         y = batch.to(device)

#     out = model(y)
#     print("Print de control: Salida media del modelo:", out.mean().item())

# x_hat = model(y)
# loss_val = loss_fn(x_hat, y, physics=physics, model=model)  # o loss_fn(x_hat, y, physics) según tu get_loss
# print("Loss prueba:", loss_val.mean().item())
# for name, p in model.named_parameters():
#     if p.grad is None:
#         print(name, "NO tiene grad")
#     else:
#         print(name, p.grad.abs().mean())
# x = x.to(device)
# out = model(x)
# print(out.requires_grad)  # ¿True?
# loss = loss_fn(out, y, physics=physics, model=model)
# print(loss.requires_grad) # ¿True?
# with torch.no_grad():
#     out = model(x)