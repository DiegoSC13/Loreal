
from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
# from tqdm import tqdm
import deepinv as dinv

from dataset import LorealDataset, FastDVDnetDataset
from model import FastDVDnet
from losses import get_loss
from physics import get_physics

from pathlib import Path
import os

from torchvision import transforms, datasets

from deepinv.utils import get_data_home
from deepinv.models.utils import get_weights_url

# importa tu modelo
from model import FastDVDnet
from losses import get_loss

from datetime import datetime
import os


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

parser.add_argument("--batch_size", type=int, default=32) #Default of SSIBench, not sure if it's the best option here
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

# =========================
# CONFIG
# =========================

# BASE_DIR = Path(".")
# DATA_DIR = BASE_DIR / "input_images"
# CKPT_DIR = BASE_DIR / "ckpts"

# Set the global random seed and select device
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Creo directorio con fechas para no sobreescribir
timestamp = datetime.now().strftime("%y-%m-%d_%H_%M_%S")
save_path = os.path.join(args.output_path, f"run_{timestamp}")
os.makedirs(save_path, exist_ok=True)

#Checkpoints dir
ckpt_path = os.path.join(save_path, "ckpts") 
os.makedirs(ckpt_path, exist_ok=True)

# Memory/efficiency limitations
batch_size = args.batch_size if torch.cuda.is_available() else 1

# Dataset preparation
patch_size = tuple(args.patch_size) if args.patch_size else None
base_dirs = [
    "/mnt/bdisk/dewil/loreal_POC2/sequences_for_self-supervised_tests/easier",
    "/mnt/bdisk/dewil/loreal_POC2/sequences_for_self-supervised_tests/difficult"
    # "/Users/diegosilveracoeff/Desktop/PhD/data/easier"
]
# dataset = LorealDataset(image_paths=args.image_paths,
#                         transform=args.transform,
#                         patch_size=patch_size)
 
dataset = FastDVDnetDataset(
    base_dirs=base_dirs,
    patch_size=patch_size
)

stack, target = dataset[0]
print(f"Stack shape: {stack.shape}, Target shape: {target.shape}\n")

# Dataloader preparation
n_train = int(0.8 * len(dataset))
n_test = len(dataset) - n_train
train_dataset, test_dataset = random_split(dataset, [n_train, n_test])

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
test_dataloader  = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

for x, target in train_dataloader:
    print("x shape after dataloader:", x.shape)
    print("Target shape after dataloader:", target.shape)
    break  # solo miro el primer batch

# Choose model
model=FastDVDnet(num_input_frames=5).to(device)

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
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma) #Quiero multiplicar el lr por 0.1 cada 50 epocas
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(args.epochs * 0.8) + 1)

# start with a pretrained model to reduce training time
if args.ckpt:
    ckpt = torch.load(args.ckpt, map_location=device)

    state_dict = ckpt.get("state_dict", ckpt)
    scheduler.load_state_dict(ckpt["scheduler"])

    # Adaptar primera conv si hace falta, ASK VALERY
    for key in list(state_dict.keys()):
        if "inc.convblock.0.weight" in key and state_dict[key].shape[1] == 1:
            w = state_dict[key]
            state_dict[key] = w.repeat(1, 3, 1, 1) / 3

    model.load_state_dict(state_dict, strict=False)

    print("Loaded model weights (optimizer reinitialized).")

verbose = True  # print training information

for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0
    
    for i, (stack, target) in enumerate(train_dataloader):
        stack = stack.to(device)   # [B, 5, H, W]
        target = target.to(device) # [B, 1, H, W]
        
        optimizer.zero_grad()
        output = model(stack)      # [B, 1, H, W]

       # Loss DeepInverse
        loss = loss_fn(output, target, physics, model)
        loss = loss.mean()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    # Para actualizar lr
    scheduler.step()

    print(
    f"Epoch {epoch+1}, "
    f"Loss: {running_loss/len(train_dataloader):.6f}, "
    f"lr: {optimizer.param_groups[0]['lr']:.2e}\n"
)
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

model.eval()  # importante
with torch.no_grad():
    batch = next(iter(train_dataloader))  # toma el primer batch
    if isinstance(batch, (tuple, list)):
        y = batch[0].to(device)  # ajusta según tu dataset
    else:
        y = batch.to(device)

    out = model(y)
    print("Print de control: Salida media del modelo:", out.mean().item())

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