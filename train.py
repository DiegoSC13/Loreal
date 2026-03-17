
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

import torch
from torch.utils.data import DataLoader, random_split

from dataset import LorealDataset, FastDVDnetDataset
from model import FastDVDnet, SureWrapper
from losses import get_loss
from physics import get_physics

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
args = parser.parse_args()

#Creo directorio con fechas para no sobreescribir
timestamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
save_path = os.path.join(args.output_path, f"train_{timestamp}")
os.makedirs(save_path, exist_ok=True)

log_path = os.path.join(args.output_path, f"log_{timestamp}.log")

logger = logging.getLogger(__name__)
logger.info(f"Python version: {platform.python_version()}")

# Esto genera el comando completo tal como se ejecutó
command_used = " ".join(sys.argv)

command_used = shlex.join(sys.argv)
#logger.info(args)
logger.info(f"Command used: {command_used}")

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
subdirs = []
for root, dirs, files in os.walk(root_dir, followlinks=True):
    # excluir check del recorrido
    dirs[:] = [d for d in dirs if d != "check"]
    for d in dirs:
        subdirs.append(os.path.join(root, d))

base_dirs = [root_dir] 
print(f'{base_dirs=}')


dataset = FastDVDnetDataset(
    base_dirs=base_dirs,
    patch_size=patch_size
)

stack, target = dataset[0]
print(f"Stack shape: {stack.shape}, Target shape: {target.shape}\n")
stack = stack.to(device)
target = target.to(device)

# Dataloader preparation
n_train = int(0.8 * len(dataset))
n_test = len(dataset) - n_train
train_dataset, test_dataset = random_split(dataset, [n_train, n_test], generator=generator) #generator para reproducibilidad

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
test_dataloader  = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

#Para testing
test_indices = test_dataset.indices
np.savetxt(os.path.join(save_path, "test_indices.txt"), test_indices, fmt="%d")

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
        # mc_iter=args.mc_iter
        step_size = args.step_size,  #UNSURE
        momentum  = args.momentum   #UNSURE
    )

# choose optimizer and scheduler
optimizer = torch.optim.Adam(wrapper.parameters(), lr=args.lr, weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma) #Quiero multiplicar el lr por 0.1 cada 50 epocas

if args.ckpt:
    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)  # compatible con ambos formatos
    model.load_state_dict(state_dict, strict=False)
    check_checkpoint_loading_with_magnitude(model, state_dict)
    print("Loaded model weights (optimizer reinitialized).")

epoch_losses = []
for epoch in range(args.epochs):
    wrapper.train()
    running_loss = 0.0
    # Antes del loop de batches:
    grad_accum = {name: 0.0 for name, p in model.named_parameters() if p.requires_grad}
    n_batches = 0
    for i, (stack, target) in enumerate(train_dataloader):
        stack = stack.to(device)           # [B, 5, H, W]
        y_central = stack[:, 2:3, :, :]   # [B, 1, H, W]

        wrapper.set_context(stack)         # detach interno, una sola vez
        optimizer.zero_grad()

        output = wrapper(y_central)        # forward
        loss = loss_fn(y_central, output, physics, wrapper).mean()
        loss.backward()
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

    epoch_loss = running_loss / len(train_dataloader)
    epoch_losses.append(epoch_loss)
    scheduler.step()    
    for name, avg_grad in grad_accum.items():
        print(f"{name} | grad_mean_epoch={avg_grad / n_batches:.2e}")
    print(
    f"Epoch {epoch+1}, "
    f"Loss: {epoch_loss:.8f}, "
    f"lr: {optimizer.param_groups[0]['lr']:.2e}\n"
    )
    logger.info(f"Epoch {epoch+1}/{args.epochs} - Loss: {epoch_loss:.4f} - Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

    #Printeo cada 10 epochs
    if (epoch+1) % 10 == 0: 
        ckpt_path = os.path.join(save_path, f"epoch_{epoch+1}.pth")
        torch.save({
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "loss": running_loss/len(train_dataloader)
        }, ckpt_path)

plt.figure()
plt.plot(epoch_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.show()
plt.savefig(os.path.join(save_path, "training_loss.png"), dpi=200)

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