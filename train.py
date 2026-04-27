
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

from dataset import LorealDataset, FastDVDnetDataset, get_valid_sequences, FMDDDataset, get_fmdd_sequences
from new_model import FastDVDnet_, SureWrapper
from losses import get_loss
from physics import get_physics
from deepinv.loss.r2r import R2RLoss, R2RModel
from utils import *
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- CARGA DE RUTAS LOCALES ---
def load_local_paths():
    config_path = os.path.join(os.path.dirname(__file__), 'config.sh')
    if os.path.exists(config_path):
        env_vars = {}
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    value = value.strip().strip('"').strip("'")
                    # Expand variables like ${WORKDIR} or $WORKDIR
                    for k, v in env_vars.items():
                        value = value.replace(f'${{{k}}}', v)
                        value = value.replace(f'${k}', v)
                    env_vars[key] = value
                    
                    if key == 'EXTERNAL_CODES_DIR':
                        if value not in sys.path:
                            sys.path.append(value)

load_local_paths()
import tifffile
from models_FastDVDnet_sans_noise_map import FastDVDnet

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
parser.add_argument("--sequence_directory", type=str, required=True,
                    help="Ruta al directorio que contiene las secuencias de imágenes")
parser.add_argument("--dataset_type", type=str, choices=("loreal", "fmdd"), default="loreal",
                    help="Tipo de dataset a utilizar")
parser.add_argument("--fmdd_mode", type=str, choices=("raw", "synthetic"), default="raw",
                    help="Modo para el dataset FMDD: 'raw' usa ruido real, 'synthetic' genera ruido sobre GT")
parser.add_argument("--fmdd_modalities", type=str, nargs="+", default=None,
                    help="Lista de modalidades de FMDD a incluir (solo si dataset_type=fmdd)")
parser.add_argument("--output_path", type=str, required=True, help="Directorio donde se crea carpeta de checkpoints")
parser.add_argument("--ckpt", type=str, default=None, help="Ruta a checkpoint preentrenado") # Pretrained model path
parser.add_argument("--loss", type=str, choices=("sure", "pure", "pgure", "unsure", "unpgure", "r2r_g", "r2r_p"), required=True) # "noise2score",# "unsure")
parser.add_argument("--sigma", type=float, default=None) #Gaussian std
parser.add_argument("--gamma", type=float, default=None) #Poisson scalar factor
parser.add_argument("--tau1", type=float, default=1e-3, help="Approximation constant on Monte Carlo divergence estimation") #1e-3 default value for range [0,1] 
parser.add_argument("--tau2", type=float, default=1e-2, help="Approximation constant for the second derivative") #1e-2 default value for range [0,1] 
parser.add_argument("--alpha", type=float, default=0.15) #R2R recorruption factor
# parser.add_argument("--mc_iter", type=int, default=1) #Era para el estimador de Monte Carlo de la divergencia, pero no veo que lo usen en las losses SURE-based
parser.add_argument("--step_size", type=float, nargs=2, default=(1e-5, 1e-5), help="Gradient step size") #UNSURE and PG-UNSURE 
parser.add_argument("--momentum", type=float, nargs=2, default=(0.9, 0.9), help="Gradient momentum")    #UNSURE and PG-UNSURE 

parser.add_argument("--batch_size", type=int, default=32) #Default of SSIBench is 32, not sure if it's the best option here
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--scheduler_step_size", type=int, default=100)
parser.add_argument("--scheduler_gamma", type=float, default=0.5)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--patch_size", type=int, nargs=2, default=None,
                    help="Tamaño del recorte aleatorio, ej: --patch_size 128 128")
parser.add_argument("--transform", type=str, default=None,
                    help="Nombre de la transformación a aplicar (opcional)")
parser.add_argument("--loss_scaler", type=float, default=1.0, help="Factor para escalar la pérdida antes del backward (para evitar vanishing gradients en escalas pequeñas)")
parser.add_argument("--data_scale", type=float, default=9000.0, help="Factor divisor para los datos tras la transformacion lineal (a,b)")
parser.add_argument("--patience", type=int, default=0, help="Patience for early stopping. If 0 or less, early stopping is disabled.")
args = parser.parse_args()

#Seed
seed=43
set_seed(seed)

# --- Escalado de hiperparámetros ---
# Si los datos se dividen por data_scale, los hiperparámetros de ruido deben escalarse proporcionalmente
# sigma_scaled = 1.4 * args.sigma / args.data_scale if args.sigma is not None else None
# gamma_scaled = args.gamma * 1.4 / args.data_scale if args.gamma is not None else None
sigma_scaled = args.sigma / args.data_scale if args.sigma is not None else None
gamma_scaled = args.gamma / args.data_scale if args.gamma is not None else None
tau1_scaled   = args.tau1 #/ args.data_scale 
tau2_scaled   = args.tau2 #/ args.data_scale 

print(f"\n--- Configuración de Escala (Data Scale: {args.data_scale}) ---")
if sigma_scaled is not None:
    print(f"  Sigma: {args.sigma} (Original) -> {sigma_scaled:.6f} (Escalado)")
if gamma_scaled is not None:
    print(f"  Gamma: {args.gamma} (Original) -> {gamma_scaled:.6f} (Escalado)")
print(f"  Tau1:   {args.tau1} (Original) -> {tau1_scaled:.6f} (Escalado)")
print(f"  Tau2:   {args.tau2} (Original) -> {tau2_scaled:.6f} (Escalado)")
print("---------------------------------------------------\n")

# Creo directorio con fechas para no sobreescribir
# timestamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
# save_path = os.path.join(args.output_path, f"train_{timestamp}")
# os.makedirs(save_path, exist_ok=True)
os.makedirs(args.output_path, exist_ok=True)
save_path = args.output_path

log_path = os.path.join(args.output_path, f"logfile.log")

# Subdirectorios para organizar resultados
losses_dir = os.path.join(save_path, "losses")
sequences_dir = os.path.join(save_path, "sequences")
os.makedirs(losses_dir, exist_ok=True)
os.makedirs(sequences_dir, exist_ok=True)

logger = logging.getLogger(__name__)
#logger.info(f"Python version: {platform.python_version()}")
print(f"Python version: {platform.python_version()}")
print(f"Command used: {shlex.join(sys.argv)}")

# Set the global random seed and select device
# set_seed already handled global torch seed
generator = torch.Generator().manual_seed(seed) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Checkpoints dir
ckpt_path = os.path.join(save_path, "ckpts") 
os.makedirs(ckpt_path, exist_ok=True)

# Memory/efficiency limitations
batch_size = args.batch_size if torch.cuda.is_available() else 1

# Dataset preparation
patch_size = tuple(args.patch_size) if args.patch_size else None

root_dir = args.sequence_directory

if args.dataset_type == "loreal":
    all_sequences_paths = []
    for root, dirs, files in os.walk(root_dir, followlinks=True):
        # excluir check del recorrido
        dirs[:] = [d for d in dirs if d != "check"]
        for d in dirs:
            seq_path = os.path.join(root, d)
            all_sequences_paths.append(seq_path)

    # Filtrar secuencias válidas antes de dividir
    valid_sequences_info = get_valid_sequences(all_sequences_paths, out_file=os.path.join(sequences_dir, "sequences_left_out.txt"))
elif args.dataset_type == "fmdd":
    t0_discovery = datetime.now()
    valid_sequences_info = get_fmdd_sequences(root_dir, modalities=args.fmdd_modalities)
    discovery_duration = (datetime.now() - t0_discovery).total_seconds()
    print(f"Found {len(valid_sequences_info)} sequences in FMDD (Discovery time: {discovery_duration:.1f}s)")

# Shuffle and split sequences
random.seed(seed)
random.shuffle(valid_sequences_info)

n_total = len(valid_sequences_info)
n_train = int(0.8 * n_total)
n_val = n_total - n_train
#n_val = int(0.2 * n_total)
#n_test = n_total - n_train - n_val

train_info = valid_sequences_info[:n_train]
#train_info = sorted(train_info)
val_info = valid_sequences_info[n_train:]#n_train + n_val]
#val_info = sorted(val_info)
#test_info = valid_sequences_info[n_train + n_val:]

print(f"Total valid sequences: {n_total}")
print(f"Train sequences: {len(train_info)}")
print(f"Val sequences: {len(val_info)}")
#print(f"Test sequences: {len(test_info)}")

# Transform preparation
transform = None
if args.transform == "d4":
    transform = RandomD4()
    print("Using D4 data augmentation.")

if args.dataset_type == "loreal":
    train_dataset = FastDVDnetDataset(sequence_info=train_info, patch_size=patch_size, data_scale=args.data_scale, transform=transform)
    val_dataset   = FastDVDnetDataset(sequence_info=val_info, patch_size=patch_size, data_scale=args.data_scale)
elif args.dataset_type == "fmdd":
    train_dataset = FMDDDataset(sequence_info=train_info, patch_size=patch_size, data_scale=args.data_scale, transform=transform, mode=args.fmdd_mode, gamma=args.gamma)
    val_dataset   = FMDDDataset(sequence_info=val_info, patch_size=patch_size, data_scale=args.data_scale, mode=args.fmdd_mode, gamma=args.gamma)
#test_dataset = FastDVDnetDataset(sequence_info=test_info, patch_size=patch_size, data_scale=args.data_scale)

print(f"Train dataset size (stacks): {len(train_dataset)}")
print(f"Val dataset size (stacks): {len(val_dataset)}")
#print(f"Test dataset size (stacks): {len(test_dataset)}")

# Dataloader preparation
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
val_dataloader   = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
# test_dataloader  = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

# Guardar secuencias para referencia
with open(os.path.join(sequences_dir, "train_sequences.txt"), "w") as f:
    for s_info in train_info:
        if isinstance(s_info, dict): # FMDD
            path = f"{s_info['modality']}/{s_info['seq_id']}"
        else: # Loreal
            path = s_info[0] if isinstance(s_info, (tuple, list)) else str(s_info)
        f.write(f"{path}\n")

with open(os.path.join(sequences_dir, "val_sequences.txt"), "w") as f:
    for s_info in val_info:
        if isinstance(s_info, dict): # FMDD
            path = f"{s_info['modality']}/{s_info['seq_id']}"
        else: # Loreal
            path = s_info[0] if isinstance(s_info, (tuple, list)) else str(s_info)
        f.write(f"{path}\n")

# with open(os.path.join(save_path, "test_sequences.txt"), "w") as f:
#     for s_info in test_info:
#         f.write(f"{s_info[0]}\n")

# First batch load timing
t0_first = datetime.now()
for x, target in train_dataloader:
    first_duration = (datetime.now() - t0_first).total_seconds()
    print(f"First batch load time: {first_duration:.1f}s")
    print(f"x shape: {x.shape}, target shape: {target.shape}")
    break # Just one batch to warm up and verify

# Choose model
model = FastDVDnet(num_input_frames=5).to(device)
# Wrapper que adapta FastDVDnet (5 frames -> 1 frame) a la interfaz que esperan las losses de DeepInverse
wrapper = SureWrapper(model).to(device)

# Choose physics
physics = get_physics(
    loss_name=args.loss,
    sigma=sigma_scaled,
    gamma=gamma_scaled,
    device=device
)
# Choose loss 
loss_fn = get_loss(
        loss_name=args.loss,
        device=device,
        sigma=sigma_scaled,
        gamma=gamma_scaled, 
        tau1=tau1_scaled,
        tau2=tau2_scaled,
        alpha=args.alpha,
        # mc_iter=args.mc_iter
        step_size = args.step_size,  #UNSURE
        momentum  = args.momentum   #UNSURE
        #eval_n_samples = 25          # More samples for stable evaluation
    )

# Adapt model if using R2RLoss, neccesary for R2R
if isinstance(loss_fn, R2RLoss):
    wrapper = loss_fn.adapt_model(wrapper)
    if args.loss == "r2r_p" and gamma_scaled is not None:
        print(f"\n> [R2R Poisson] Gamma (gain) configurada (escalada): {gamma_scaled:.12f}")
        print(f"> [R2R Poisson] Data scale (divisor): {args.data_scale}")
        print(f"> [R2R Poisson] IMPORTANTE: La ganancia ha sido adaptada a la escala [0, 1] usada por el modelo.\n")

# choose optimizer and scheduler
optimizer = torch.optim.Adam(wrapper.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma) #Quiero multiplicar el lr por 0.1 cada 50 epocas

if args.ckpt:
    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)  # compatible con ambos formatos
    model.load_state_dict(state_dict, strict=False)
    # check_checkpoint_loading_with_magnitude(model, state_dict)
    print("Loaded model weights (optimizer reinitialized).")

class EarlyStopping:
    def __init__(self, patience=40, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

early_stopping = EarlyStopping(patience=args.patience) if args.patience > 0 else None
best_val_loss = float('inf')
best_epoch = 0

# --- EVALUACIÓN EPOCH 0 (Punto de partida) ---
print("Evaluating Epoch 0 (baseline)...")
wrapper.eval()
initial_loss = 0.0
num_batches_eval = 0
with torch.no_grad():
    for i, (stack, target) in enumerate(train_dataloader):
        if i >= 20: break  # Usamos 20 batches para una mejor estimación
        stack = stack.to(device)
        y_central = stack[:, 2:3, :, :]
        
        if isinstance(wrapper, R2RModel):
            wrapper.model.set_context(stack)
        else:
            wrapper.set_context(stack)

        if isinstance(loss_fn, R2RLoss):
            # Para una evaluación estable, promediamos la LOSS sobre múltiples muestras
            # R2RModel solo actualiza .corruption si .training=True y update_parameters=True
            n_samples = 5 # Usamos 5 muestras por batch para no ralentizar demasiado
            batch_loss = 0
            for _ in range(n_samples):
                wrapper.training = True
                output = wrapper(y_central, physics, update_parameters=True)
                wrapper.training = False
                batch_loss += loss_fn(y_central, output, physics, wrapper).mean().item()
            loss_val = batch_loss / n_samples
        else:
            output = wrapper(y_central)
            loss_val = loss_fn(y_central, output, physics, wrapper).mean().item()

        initial_loss += loss_val
        num_batches_eval += 1
        del output, stack, y_central

avg_init_loss = initial_loss / num_batches_eval

# Evaluamos también sobre el validation set para el baseline de Val Loss, PSNR y SSIM
initial_val_loss = 0.0
initial_val_psnr = 0.0
initial_val_ssim = 0.0

with torch.no_grad():
    for i, (stack, target) in enumerate(val_dataloader):
        stack = stack.to(device)
        target = target.to(device)
        y_central = stack[:, 2:3, :, :]
        
        if isinstance(wrapper, R2RModel):
            wrapper.model.set_context(stack)
        else:
            wrapper.set_context(stack)

        if isinstance(loss_fn, R2RLoss):
            n_samples = 5 
            batch_loss = 0
            for _ in range(n_samples):
                wrapper.training = True
                output = wrapper(y_central, physics, update_parameters=True)
                wrapper.training = False
                batch_loss += loss_fn(y_central, output, physics, wrapper).mean().item()
            initial_val_loss += batch_loss / n_samples
        else:
            output = wrapper(y_central)
            loss_val = loss_fn(y_central, output, physics, wrapper).mean()
            initial_val_loss += loss_val.item()
        
        if args.dataset_type == "fmdd" or (hasattr(args, "synthetic_dataset") and args.synthetic_dataset):
            mse = torch.mean((output - target) ** 2)
            if mse > 0:
                psnr = 10 * torch.log10(1.0 / mse)
                initial_val_psnr += psnr.item()
                
                out_np = output.detach().cpu().squeeze().numpy()
                gt_np = target.detach().cpu().squeeze().numpy()
                s_val = ssim_func(gt_np, out_np, data_range=1.0)
                initial_val_ssim += s_val
                
        # Liberar memoria de GPU lo antes posible para evitar OOM
        del output, stack, y_central, target

n_val_batches = len(val_dataloader)
avg_init_val_loss = initial_val_loss / n_val_batches if n_val_batches > 0 else 0
avg_init_val_psnr = initial_val_psnr / n_val_batches if n_val_batches > 0 else 0
avg_init_val_ssim = initial_val_ssim / n_val_batches if n_val_batches > 0 else 0

print(f"Epoch 0 Train Loss: {avg_init_loss:.12f}, Val Loss: {avg_init_val_loss:.12f}, PSNR: {avg_init_val_psnr:.2f}dB, SSIM: {avg_init_val_ssim:.4f}\n")
torch.cuda.empty_cache()

epoch_losses = [avg_init_loss]
val_losses = [avg_init_val_loss]
val_psnrs = [avg_init_val_psnr]
val_ssims = [avg_init_val_ssim]
for epoch in range(args.epochs):
    epoch_start_time = datetime.now()
    wrapper.train()
    running_loss = 0.0
    # Antes del loop de batches:
    grad_accum = {name: 0.0 for name, p in model.named_parameters() if p.requires_grad}
    n_batches = 0
    t0_epoch = datetime.now()
    t_data_total = 0.0
    t_start_fetch = datetime.now()
    
    for i, (stack, target) in enumerate(train_dataloader):
        t_data_total += (datetime.now() - t_start_fetch).total_seconds()
        
        # Guardar imagen de control (solo una vez en la vida del entrenamiento)
        if epoch == 0 and i == 0:
            debug_dir = os.path.join(args.output_path, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            # Guardamos el frame central ruidoso [C, H, W] -> [H, W]
            tifffile.imwrite(os.path.join(debug_dir, "control_noisy.tif"), stack[0, 2].cpu().numpy())
            tifffile.imwrite(os.path.join(debug_dir, "control_gt.tif"), target[0, 0].cpu().numpy())
            print(f"Control images saved to {debug_dir}")
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
        t_start_fetch = datetime.now() # Reset para medir espera del siguiente batch

    # # Diagnóstico de gradientes (una vez por época, último batch)
    # for name, param in model.named_parameters():
    #     if param.requires_grad and param.grad is not None:
    #         print(f"{name} | grad={param.grad.abs().mean():.2e} | val={param.abs().mean():.2e}")
    train_duration = datetime.now() - t0_epoch
    # Validation loop
    val_t0 = datetime.now()
    wrapper.eval()
    val_loss = 0.0
    val_psnr = 0.0
    val_ssim = 0.0
    with torch.no_grad():
        for i, (stack, target) in enumerate(val_dataloader):
            stack = stack.to(device)
            target = target.to(device)
            y_central = stack[:, 2:3, :, :]
            
            if isinstance(wrapper, R2RModel):
                wrapper.model.set_context(stack)
            else:
                wrapper.set_context(stack)

            if isinstance(loss_fn, R2RLoss):
                # Evaluación estable promediando LOSS sobre múltiples muestras
                n_samples = 5 
                batch_loss = 0
                for _ in range(n_samples):
                    wrapper.training = True
                    output = wrapper(y_central, physics, update_parameters=True)
                    wrapper.training = False
                    batch_loss += loss_fn(y_central, output, physics, wrapper).mean().item()
                val_loss += batch_loss / n_samples
            else:
                output = wrapper(y_central)
                loss = loss_fn(y_central, output, physics, wrapper).mean()
                val_loss += loss.item()
            
            # Cálculo de PSNR (asumiendo que target es Ground Truth)
            # Solo tiene sentido si target es realmente limpio (como en FMDD o modo sintético)
            if args.dataset_type == "fmdd" or (hasattr(args, "synthetic_dataset") and args.synthetic_dataset):
                mse = torch.mean((output - target) ** 2)
                if mse > 0:
                    psnr = 10 * torch.log10(1.0 / mse)
                    val_psnr += psnr.item()
                    
                    # Compute SSIM
                    out_np = output.detach().cpu().squeeze().numpy()
                    gt_np = target.detach().cpu().squeeze().numpy()
                    s_val = ssim_func(gt_np, out_np, data_range=1.0)
                    val_ssim += s_val
            
            del output, stack, y_central, target
    
    torch.cuda.empty_cache()
    val_duration = datetime.now() - val_t0

    n_val_batches = len(val_dataloader)
    avg_val_loss = val_loss / n_val_batches if n_val_batches > 0 else 0
    avg_val_psnr = val_psnr / n_val_batches if n_val_batches > 0 else 0
    avg_val_ssim = val_ssim / n_val_batches if n_val_batches > 0 else 0
    val_losses.append(avg_val_loss)
    val_psnrs.append(avg_val_psnr)
    val_ssims.append(avg_val_ssim)

    epoch_loss = running_loss / len(train_dataloader)
    epoch_losses.append(epoch_loss)
    scheduler.step()    
    
    # Save current loss history
    np.save(os.path.join(losses_dir, "train_losses.npy"), np.array(epoch_losses))
    np.save(os.path.join(losses_dir, "val_losses.npy"), np.array(val_losses))
    
    print(
    f"Epoch {epoch+1}, "
    f"Loss: {epoch_loss:.12f}, Val Loss: {avg_val_loss:.12f}, "
    f"PSNR/SSIM: {avg_val_psnr:.2f}dB / {avg_val_ssim:.4f}, "
    f"lr: {optimizer.param_groups[0]['lr']:.2e}\n"
    f"Time: Train: {train_duration.total_seconds():.1f}s (Data wait: {t_data_total:.1f}s), Val: {val_duration.total_seconds():.1f}s\n"
    )
    logger.info(f"Epoch {epoch+1}/{args.epochs} - Loss: {epoch_loss:.12f} - Val Loss: {avg_val_loss:.12f} - PSNR: {avg_val_psnr:.2f} - SSIM: {avg_val_ssim:.4f} - LR: {optimizer.param_groups[0]['lr']:.2e}")

    if (epoch+1) % 10 == 0: 
        this_ckpt_path = os.path.join(ckpt_path, f"epoch_{epoch+1}.pth")
        torch.save({
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "loss": running_loss/len(train_dataloader)
        }, this_ckpt_path)

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch + 1
        torch.save({
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "loss": avg_val_loss
        }, os.path.join(ckpt_path, "best_model.pth"))

    # Early stopping check
    if early_stopping is not None:
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

print(f"Mejor epoca (best_model.pth): {best_epoch}")

# Save final model state (in case early stopping happened or EPOCHS was not a multiple of 10)
final_ckpt_path = os.path.join(ckpt_path, f"epoch_{epoch+1}.pth")
if not os.path.exists(final_ckpt_path):
    torch.save({
        "epoch": epoch + 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "loss": running_loss/len(train_dataloader)
    }, final_ckpt_path)

plt.figure(figsize=(10, 6))
plt.plot(epoch_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

# Robust scaling logic for SURE losses (can be negative)
all_losses = np.concatenate([epoch_losses, val_losses])
min_l, max_l = np.min(all_losses), np.max(all_losses)

if min_l > 0 and (max_l / (min_l + 1e-12) > 100):
    plt.yscale('log')
    plt.title("Training vs Validation Loss (Log Scale)")
elif min_l < 0:
    # If there are negative values, log scale is not possible. 
    # We use symlog if the range is large, or linear if it's manageable.
    if max_l - min_l > 1000:
        plt.yscale('symlog', linthresh=1.0)
        plt.title("Training vs Validation Loss (SymLog Scale)")
    else:
        plt.title("Training vs Validation Loss (Linear Scale)")
else:
    plt.title("Training vs Validation Loss (Linear Scale)")

plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.savefig(os.path.join(losses_dir, "loss_plot.png"), dpi=200)
# plt.show() # Commented to avoid blocking in non-interactive environments

# Also save as a text file for easy reading
with open(os.path.join(losses_dir, "losses.txt"), "w") as f:
    f.write("Epoch, TrainLoss, ValLoss, ValPSNR, ValSSIM\n")
    for i, (tl, vl, vp, vs) in enumerate(zip(epoch_losses, val_losses, val_psnrs, val_ssims)):
        f.write(f"{i}, {tl:.12f}, {vl:.12f}, {vp:.4f}, {vs:.6f}\n")

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