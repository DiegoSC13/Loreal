import os
import sys
from os.path import dirname, join
import argparse
import time
import numpy as np

import torch
import torch.nn as nn

# --- CARGA DE RUTAS LOCALES ---
def load_local_paths():
    config_path = os.path.join(os.path.dirname(__file__), 'env_paths.sh')
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

from models_FastDVDnet_sans_noise_map import FastDVDnet
import iio
import tifffile
from functions import *
from physics import get_physics

# CUDA check
cuda = torch.cuda.is_available()
if cuda:
    device = torch.device('cuda')
    print("CUDA is available")

   
def apply_tta(x, mode):
    """
    Applies one of 8 geometric transformations to the input tensor.
    x: [B, C, H, W]
    mode: 0 to 7
    """
    if mode == 0:
        return x
    elif mode == 1:
        return torch.rot90(x, 1, [2, 3])
    elif mode == 2:
        return torch.rot90(x, 2, [2, 3])
    elif mode == 3:
        return torch.rot90(x, 3, [2, 3])
    elif mode == 4:
        return torch.flip(x, [3])
    elif mode == 5:
        return torch.flip(torch.rot90(x, 1, [2, 3]), [3])
    elif mode == 6:
        return torch.flip(torch.rot90(x, 2, [2, 3]), [3])
    elif mode == 7:
        return torch.flip(torch.rot90(x, 3, [2, 3]), [3])
    return x

def inv_tta(y, mode):
    """
    Applies the inverse of the geometric transformation to the output tensor.
    y: [B, C, H, W]
    mode: 0 to 7
    """
    if mode == 0:
        return y
    elif mode == 1:
        return torch.rot90(y, -1, [2, 3])
    elif mode == 2:
        return torch.rot90(y, -2, [2, 3])
    elif mode == 3:
        return torch.rot90(y, -3, [2, 3])
    elif mode == 4:
        return torch.flip(y, [3])
    elif mode == 5:
        return torch.rot90(torch.flip(y, [3]), -1, [2, 3])
    elif mode == 6:
        return torch.rot90(torch.flip(y, [3]), -2, [2, 3])
    elif mode == 7:
        return torch.rot90(torch.flip(y, [3]), -3, [2, 3])
    return y

def eval(**args):
    """
    Main function
    args: Parameters
    """

    ################
    # LOAD THE MODEL
    ################
    model = FastDVDnet(5)
    model.to(device)
   
    #Load saved weights
    ckpt = torch.load(args['network'], map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    print(f"Weights loaded from {args['network']}")
    model.cuda()

    # Physics initialization for R2R recorruption
    physics = None
    if args['n_samples'] > 1:
        # Scaling parameters consistent with train.py
        sigma_scaled = args['sigma'] / args['data_scale'] if args['sigma'] is not None else None
        gamma_scaled = args['gamma'] / args['data_scale'] if args['gamma'] is not None else None
        
        physics = get_physics(
            loss_name=args['loss'],
            sigma=sigma_scaled,
            gamma=gamma_scaled,
            device=device
        )
        print(f"Physics initialized for recorruption (Loss: {args['loss']}, Samples: {args['n_samples']}, Alpha: {args['alpha']})")

    out_dir = os.path.dirname(args['output'])
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
       
    # Initialisation
    input_path_raw = args['input']
    has_template = "%" in input_path_raw
    
    if has_template:
        input_path = input_path_raw % (args['first'])
    else:
        input_path = input_path_raw

    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        print("Please check your path configuration in env_paths.sh")
        sys.exit(1)
        
    ut = iio.read(input_path)
    H, W, _ = ut.shape
    H = 4*(H//4)
    W = 4*(W//4)

    # Pre-processing data is optional (default a=1, b=0 for FMDD)
    if args['pre_processing_data'] and os.path.exists(args['pre_processing_data']):
        [a,b] = np.loadtxt(args['pre_processing_data'])
    else:
        print("No pre-processing file found. Using defaults a=1.0, b=0.0")
        a, b = 1.0, 0.0

    model.eval()

    def add_poisson_noise(img, gamma):
        """Adds synthetic Poisson noise [0, 1] matching dataset.py logic"""
        if gamma is None:
            return img
        img = torch.clamp(img, min=0.0)
        noisy = torch.poisson(img * gamma) / gamma
        return noisy

    def calculate_psnr(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0: return 100
        return 20 * torch.log10(1.0 / torch.sqrt(mse))

    psnrs = []
    
    # Si es sintético de imagen única, el rango puede ser solo [first]
    if not has_template:
        test_range = range(args['first'], args['first'] + 1)
    else:
        test_range = range(args['first']+2, args['last']-1)
    
    ns_suffix = f"_ns{args['n_samples']}" if args['n_samples'] > 1 else ""
    
    for i in test_range:
    
        if args['synthetic_test']:
            # In synthetic mode, we assume the input is clean (GT) and we add noise
            # We generate 5 realizations based on the central frame's intensity
            path = args['input'] % (i) if has_template else args['input']
            ut = reads_image(path, H, W, im_range=1)[:1]
            frames = [add_poisson_noise(ut, args['gamma']) for _ in range(5)]
            
            # Siempre guardamos el ruidoso de entrada para comparar en modo sintético
            noisy_out = frames[2].cpu().numpy().squeeze()
            tifffile.imwrite(os.path.join(out_dir, f"input_noisy{ns_suffix}_{i:03d}.tif"), noisy_out)
        else:
            ut_moins_2 = reads_image(args['input']%(i-2), H, W, im_range=1)[:1]
            ut_moins_1 = reads_image(args['input']%(i-1), H, W, im_range=1)[:1]
            ut         = reads_image(args['input']%(i)  , H, W, im_range=1)[:1]
            ut_plus_1  = reads_image(args['input']%(i+1), H, W, im_range=1)[:1]
            ut_plus_2  = reads_image(args['input']%(i+2), H, W, im_range=1)[:1]
            frames = [ut_moins_2, ut_moins_1, ut, ut_plus_1, ut_plus_2]
            
            if args['save_noisy']:
                noisy_out = ut.cpu().numpy().squeeze()
                tifffile.imwrite(os.path.join(out_dir, f"input_noisy{ns_suffix}_{i:03d}.tif"), noisy_out)

        stack = torch.stack(frames, dim=0).contiguous().view((1, 5, H, W)).cuda()

        stack = linear_transform(stack, a, b, u=1) / args['data_scale'] # The frames can be in the range [0,9000]. here, we just normalize them to the range [0,1]

        with torch.no_grad():
            n_samples = args['n_samples']
            n_geom = 8 if args['geometric_ensemble'] else 1
            out_sum = torch.zeros((1, 1, H, W)).cuda()

            for _ in range(n_samples):
                # Recorrupt if needed (R2R)
                if n_samples > 1 and physics is not None:
                    # Recorrupt only the central frame (index 2)
                    y_central = stack[:, 2:3, :, :].clone()
                    y_central = torch.clamp(y_central, min=1e-6)
                    
                    if hasattr(physics.noise_model, 'generate_noise'):
                        noise = physics.noise_model.generate_noise(y_central)
                    else:
                        noise = physics.noise_model(y_central) - y_central
                        
                    y_recorrupted = y_central + args['alpha'] * noise
                    current_stack = stack.clone()
                    current_stack[:, 2:3, :, :] = y_recorrupted
                else:
                    current_stack = stack

                # Geometric TTA loop
                for m in range(n_geom):
                    transformed_stack = apply_tta(current_stack, m)
                    y_out = model(transformed_stack)
                    out_sum += inv_tta(y_out, m)
            
            out = out_sum / (n_samples * n_geom)

        out = linear_transform(out*args['data_scale'], a, b, u=1, inverse=True) # before inversing the linear transform, we unnormalize them back to the big range [0,9000]

        if args['synthetic_test']:
            current_psnr = calculate_psnr(ut.cuda(), out)
            psnrs.append(current_psnr.item())
            print(f"Frame {i:03d} | PSNR: {current_psnr.item():.2f} dB")

        out = out.detach().cpu().numpy().squeeze()

        print("Frame = {:02d}".format(i))
            
        # store the result
        base, ext = os.path.splitext(args['output'])
        out_path = (base + ns_suffix + ext) % i
        tifffile.imwrite(out_path, out)

    if args['synthetic_test'] and psnrs:
        avg_psnr = sum(psnrs) / len(psnrs)
        print("-" * 30)
        print(f"PSNR Promedio: {avg_psnr:.2f} dB")
        print("-" * 30)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Blind_denoising_grayscale")
    parser.add_argument("--input"              , type=str, default=""           , help='path to input frames (C type)'                                               )
    parser.add_argument("--output"             , type=str, default="./%03d.png" , help='path to output image (C type)'                                               )
    parser.add_argument("--pre_processing_data", type=str, default=""           , help='path to the pre_processing txt file that contains the noise curve parameters')
    parser.add_argument("--first"              , type=int, default=1            , help='index first frame'                                                           ) 
    parser.add_argument("--last"               , type=int, default=40           , help='index last frame'                                                            )
    parser.add_argument("--network"            , type=str, default="./model.pth", help='path to the network'                                                         )
    parser.add_argument("--data_scale"         , type=float, default=9000.0     , help='factor divisor for data normalization'                                       )
    
    # R2R Self-Ensemble arguments
    parser.add_argument("--n_samples"          , type=int, default=1            , help='number of noise realizations to average (1 = no ensemble)'                   )
    parser.add_argument("--alpha"              , type=float, default=0.15       , help='recorruption factor used during training'                                    )
    parser.add_argument("--loss"               , type=str, default="r2r_p"      , choices=("r2r_g", "r2r_p"), help='loss type (to select noise model)'               )
    parser.add_argument("--sigma"              , type=float, default=None       , help='Gaussian std (original scale)'                                               )
    parser.add_argument("--gamma"              , type=float, default=None       , help='Poisson gain (original scale)'                                               )
    parser.add_argument("--geometric_ensemble" , action='store_true'             , help='enable geometric TTA (rotations and flips)'                                  )
    parser.add_argument("--synthetic_test"     , action='store_true'             , help='inject synthetic Poisson noise into clean input'                             )
    parser.add_argument("--save_noisy"         , action='store_true'             , help='save the noisy realization for comparison'                                   )

    argspar = parser.parse_args()

    eval(**vars(argspar))
