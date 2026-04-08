#It's basically Valery's test file, I just read the .pth file differently

import os
from os.path import dirname, join
import argparse
import time
import numpy as np

import torch
import torch.nn as nn

import sys
#sys.path.append("/mnt/bdisk/dewil/loreal_POC2/sequences_for_self-supervised_tests/FastDVDnet_codes")
sys.path.append("/home/diegosilvera/Descargas")
from models_FastDVDnet_sans_noise_map import FastDVDnet

import iio

from utils import linear_transform

def reads_image(path, H, W, im_range=255):
    import iio
    image = iio.read(path)[:H, :W]
    image = image / im_range
    if len(image.shape) == 2: # grayscale
        image = np.expand_dims(image, axis=0) # [1, H, W]
    else:
        # If it's [H, W, C], transpose to [C, H, W]
        image = image.transpose(2,0,1)
    return torch.Tensor(image)

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
if cuda:
    print("CUDA is available")

   
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
   
    # #Load saved weights
    # state_dict = torch.load(args['network'], weights_only=True)
    # model.load_state_dict(state_dict)
    # model.cuda()
    ckpt = torch.load(args['network'], map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    # Create output directory if it doesn't exist
    out_dir = os.path.dirname(args['output'])
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
       
    #Initialisation
    ut = iio.read(args['input'] % (args['first']))
    H, W, _ = ut.shape
    H = 4*(H//4)
    W = 4*(W//4)
    [a,b] = np.loadtxt(args['pre_processing_data'])

    model.eval()

    for i in range(args['first']+2, args['last']-1):
    
        ut_moins_2 = reads_image(args['input']%(i-2), H, W, im_range=1)[:1]
        ut_moins_1 = reads_image(args['input']%(i-1), H, W, im_range=1)[:1]
        ut         = reads_image(args['input']%(i)  , H, W, im_range=1)[:1]
        ut_plus_1  = reads_image(args['input']%(i+1), H, W, im_range=1)[:1]
        ut_plus_2  = reads_image(args['input']%(i+2), H, W, im_range=1)[:1]

        inframes = [ut_moins_2, ut_moins_1, ut, ut_plus_1, ut_plus_2]
        stack = torch.stack(inframes, dim=0).contiguous().view((1, 5, H, W)).cuda()

        stack = linear_transform(stack, a, b) / 9000 # The frames can be in the range [0,9000]. here, we just normalize them to the range [0,1]

        with torch.no_grad():
            out = model(stack)

        out = linear_transform(out*9000, a, b, inverse=True) # before inversing the linear transform, we unnormalize them back to the big range [0,9000]

        out = out.detach().cpu().numpy().squeeze()

        print("Frame = {:02d}".format(i))
            
        #store the result
        iio.write(args['output']%i, out)



if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Blind_denoising_grayscale")
    parser.add_argument("--input"              , type=str, default=""           , help='path to input frames (C type)'                                               )
    parser.add_argument("--output"             , type=str, default="./%03d.png" , help='path to output image (C type)'                                               )
    parser.add_argument("--pre_processing_data", type=str, default=""           , help='path to the pre_processing txt file that contains the noise curve parameters')
    parser.add_argument("--first"              , type=int, default=1            , help='index first frame'                                                           ) 
    parser.add_argument("--last"               , type=int, default=40           , help='index last frame'                                                            )
    parser.add_argument("--network"            , type=str, default="./model.pth", help='path to the network'                                                         )

    argspar = parser.parse_args()

    eval(**vars(argspar))
