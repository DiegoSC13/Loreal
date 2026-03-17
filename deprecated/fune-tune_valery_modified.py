import argparse
import iio
import numpy as np
import os
from os.path import join, dirname
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from scipy.ndimage.morphology import binary_dilation

from functions import *

# from torchvision import transforms, datasets

# from deepinv.utils import get_data_home
# from deepinv.models.utils import get_weights_url

# importa tu modelo
from model import FastDVDnet
from losses import get_loss

cuda = torch.cuda.is_available()
if cuda:
    device = torch.device('cuda')
    print("CUDA is available")

#interp = 'bilinear'
interp = 'bicubic'
class Loss(nn.Module):
    fff = 0
    def __init__(self):
        super(Loss, self).__init__()
        self.criterion = nn.L1Loss(reduction = 'mean')

    def cubic_interpolation(self, A, B, C, D, x):
        a,b,c,d = A.size()
        x = x.view(a,1,c,d).repeat(1,3,1,1)
        return B + 0.5*x*(C - A + x*(2.*A - 5.*B + 4.*C - D + x*(3.*(B - C) + D - A)))


    def bicubic_interpolation(self, im, grid):
        B, C, H, W = im.size()
        assert B == 1, "For the moment, this interpolation only works for B=1."
    
        x0 = torch.floor(grid[0, 0, :, :] - 1).long()
        y0 = torch.floor(grid[0, 1, :, :] - 1).long()
        x1 = x0 + 1
        y1 = y0 + 1
        x2 = x0 + 2
        y2 = y0 + 2
        x3 = x0 + 3
        y3 = y0 + 3
    
        x0 = x0.clamp(0, W-1)
        y0 = y0.clamp(0, H-1)
        x1 = x1.clamp(0, W-1)
        y1 = y1.clamp(0, H-1)
        x2 = x2.clamp(0, W-1)
        y2 = y2.clamp(0, H-1)
        x3 = x3.clamp(0, W-1)
        y3 = y3.clamp(0, H-1)
    
        A = self.cubic_interpolation(im[:, :, y0, x0], im[:, :, y1, x0], im[:, :, y2, x0],
                                     im[:, :, y3, x0], grid[:, 1, :, :] - torch.floor(grid[:, 1, :, :]))
        B = self.cubic_interpolation(im[:, :, y0, x1], im[:, :, y1, x1], im[:, :, y2, x1],
                                     im[:, :, y3, x1], grid[:, 1, :, :] - torch.floor(grid[:, 1, :, :]))
        C = self.cubic_interpolation(im[:, :, y0, x2], im[:, :, y1, x2], im[:, :, y2, x2],
                                     im[:, :, y3, x2], grid[:, 1, :, :] - torch.floor(grid[:, 1, :, :]))
        D = self.cubic_interpolation(im[:, :, y0, x3], im[:, :, y1, x3], im[:, :, y2, x3],
                                     im[:, :, y3, x3], grid[:, 1, :, :] - torch.floor(grid[:, 1, :, :]))
        return self.cubic_interpolation(A, B, C, D, grid[:, 0, :, :] - torch.floor(grid[:, 0, :, :]))


    def warp(self, x, flow, interp='bicubic'):
        """
        Differentiably warp a tensor according to the given optical flow.
    
        Args:
            x    : torch.Tensor of dimension [B, C, H, W], image to be warped.
            flow : torch.Tensor of dimension [B, 2, H, W], optical flow
            inter: str, can be 'nearest', 'bilinear' or 'bicubic'
        
        Returns:
            y   : torch.Tensor of dimension [B, C, H, W], image warped according to flow
            mask: torch.Tensor of dimension [B, 1, H, W], mask of undefined pixels in y
        """
        B, C, H, W = x.size()
        yy, xx = torch.meshgrid(torch.arange(H, device=x.device),
                                torch.arange(W, device=x.device))
    
        xx, yy = map(lambda x: x.view(1,1,H,W), [xx,yy])
    
        grid = torch.cat((xx, yy), 1).float()
        vgrid = Variable(grid) + flow.to(x.device)
    
        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0*vgrid[:, 0, :, :]/(W-1) - 1.0
        vgrid[:, 1, :, :] = 2.0*vgrid[:, 1, :, :]/(H-1) - 1.0
        mask = (vgrid[:, 0, :, :] >= -1) * (vgrid[:, 0, :, :] <= 1) *\
               (vgrid[:, 1, :, :] >= -1) * (vgrid[:, 1, :, :] <= 1)
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid, padding_mode="border",
                                           mode=interp, align_corners=True)
    
        mask = mask.unsqueeze(1)
        return output, mask
  

    # Computes the occlusion map based on the optical flow
    def occlusion_mask(self, warped, of, old_mask):
        """
        Computes an occlusion mask based on the optical flow
        warped: [B, C, H, W] warped frame (only used for size)
        of: [B, 2, H, W] flow
        old_mask: [B, C, H, W] first estimate of the mask
        """
        B,C,H,W = warped.size() # Suppose B==1
        a = np.zeros((1,1,H,W))
        b = np.zeros((1,1,H,W))

        a[:, :, :-1, :] = (of[0, 0, 1:, :] - of[0, 0, :-1, :])
        b[:, :, :, :-1] = (of[0, 1, :, 1:] - of[0, 1, :, :-1])
        mask = np.abs(a + b) > 0.75

        if interp == 'bicubic':
            # Slighlty dilates the occlusion map to remove pixels estimated with wrong values
            # bicubic interpolation uses a 4x4 kernel
            boule = np.ones((4, 4))
            mask[0, 0, :, :] = binary_dilation(mask[0, 0, :, :], boule)

            # Remove the boundaries (values extrapolated on the boundaries)
            mask[:, :, 1, :] = 1
            mask[:, :, mask.shape[2]-2, :] = 1
            mask[:, :, :, 1] = 1
            mask[:, :, :, mask.shape[3]-2] = 1
            mask[:, :, 0, :] = 1
            mask[:, :, mask.shape[2]-1, :] = 1
            mask[:, :, :, 0] = 1
            mask[:, :, :, mask.shape[3]-1] = 1
        else:
            # Slighlty dilates the occlusion map to remove pixels estimated with wrong values
            # bilinear interpolation uses a 2x2 kernel
            boule = np.ones((2, 2))
            mask[0, 0, :, :] = binary_dilation(mask[0, 0, :, :], boule)

            # Remove the boundaries (values extrapolated on the boundaries)
            mask[:, :, 0, :] = 1
            mask[:, :, mask.shape[2]-1, :] = 1
            mask[:, :, :, 0] = 1
            mask[:, :, :, mask.shape[3]-1] = 1

        # Invert the mask because we want a mask of good pixels
        mask = torch.Tensor(1-mask).cuda()
        mask = mask.view(1,1,H,W).repeat(1,C,1,1)
        mask = old_mask * mask
        return mask
        #return torch.ones(warped.size()).cuda()

    def forward(self, input1, prev1, flow1, mask1_0, exclusive_mask1, no_warping):
        
        
        if no_warping:
            self.loss = self.criterion(input1, prev1)
        else:
            # Warp input on target
            warped1, mask1 = self.warp(input1, flow1)
            # Compute the occlusion mask
            mask1 = self.occlusion_mask(warped1, flow1, mask1)
            Mask1 = mask1_0*mask1*exclusive_mask1

            self.loss = self.criterion(Mask1*warped1, Mask1*prev1)  
        
        return self.loss



def MF2F(**args):
    """
    Main function
    args: Parameters
    """

    ################
    # LOAD THE MODEL
    ################

    try:
        model = FastDVDnet(num_input_frames=5)
        model.load_state_dict(torch.load(args['network']))
        model.to(device)
    except:
        model = torch.load(args['network'])[0].to(device)



    #################
    # DEFINE THE LOSS
    #################

    # The loss needs to be changed when used with different networks
    lr = args['lr']
    weight_decay = 0.00001
    
    criterion = Loss() 
    criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)

    #####   Useful thinks   #####
    #Initialisation

    H, W, C = args['crops_size'], args['crops_size'], 1

    
    output_path = dirname(args['output']) +"/"

    optimizer.zero_grad()

    ###########
    # MAIN LOOP
    ###########
    easier_folder_path = '/mnt/bdisk/dewil/loreal_POC2/sequences_for_self-supervised_tests/easier'
    difficult_folder_path = '/mnt/bdisk/dewil/loreal_POC2/sequences_for_self-supervised_tests/difficult'

    #list_of_seq = [seq for seq in glob(join(args['input'], '*'))]
    list_of_seq = [seq for seq in glob(join(easier_folder_path, '*'))] + [seq for seq in glob(join(difficult_folder_path, '*'))]
    nb_seqs     = len(list_of_seq)
    print("We found ",nb_seqs, " sequences")

    folder_seq=dirname(list_of_seq[0])
    len_folder_seq = len(folder_seq)+1


    for training in range(args['nb_trainings']):
        ##select randomly a sequence 
        seq = list_of_seq[np.random.randint(nb_seqs)]
        #name = seq[-25:]
        nb_frames = len(glob(join(seq, '*')))
        seq= seq[len_folder_seq:]


        #for this seq, select randomly a stack of frame center in a random index i
        i = np.random.randint(4, nb_frames-4) #index of center frame (ut use for the training)

        frame = glob(join(args['input'], seq,  '*'))[0]
        frame = frame[-16:]
        if 'c0' in frame:
            image_name = 'image_c0_%03d.tif'
            #flo_name   = 'image_c0_%03d.flo'
            #mask_name  = 'image_c0_%03d.png'
            flo_name   = '%03d.flo'
            mask_name  = '%03d.png'
        else:
            image_name = 'image_c1_%03d.tif'
            #flo_name   = 'image_c1_%03d.flo'
            #mask_name  = 'image_c1_%03d.png'
            flo_name   = '%03d.flo'
            mask_name  = '%03d.png'
        a,b = np.loadtxt(join(args['input'], seq, "pre_processing.txt"))

        ut_moins_4 = (torch.mean(reads_image(join(args['input'], seq, image_name % (i-4)), H, W, args['range']), 0)).unsqueeze(0)
        ut_moins_2 = (torch.mean(reads_image(join(args['input'], seq, image_name % (i-2)), H, W, args['range']), 0)).unsqueeze(0)
        ut         = (torch.mean(reads_image(join(args['input'], seq, image_name % (i  )), H, W, args['range']), 0)).unsqueeze(0)
        ut_plus_2  = (torch.mean(reads_image(join(args['input'], seq, image_name % (i+2)), H, W, args['range']), 0)).unsqueeze(0)
        ut_plus_4  = (torch.mean(reads_image(join(args['input'], seq, image_name % (i+4)), H, W, args['range']), 0)).unsqueeze(0)
        
        target = (torch.mean(reads_image(join(args['target'], seq, image_name % (i-1)), H, W, args['range']), 0)).unsqueeze(0).unsqueeze(0)

        #Creation of the stack
        current_h, current_w = ut_moins_4.shape[1], ut_moins_4.shape[2] 
        inframes = [ut_moins_4, ut_moins_2, ut, ut_plus_2, ut_plus_4]
        stack = torch.stack(inframes, dim=0).contiguous().view((1, 5*C, current_h, current_w)).to(device)
        stack = linear_transform(stack, a, b)
        #Use a stack of size multiple of 8
        current_h, current_w = 8*(current_h//8), 8*(current_w//8)
        stack = stack[:,:,:current_h, :current_w]
        target = target[:,:,:current_h, :current_w]
        
        if args['zero_flow']:
            flow, mask, exclusive_mask = None, None, None
        else:
            flow = gives_flow(join(args['flow'], seq, flo_name%(i-1)), current_h, current_w)
            mask, exclusive_mask = gives_masks(join(args['mask_collision'], seq, mask_name%(i-1)), join(args['mask_warping_res'], seq, mask_name%(i-1)), current_h, current_w)
            mask, exclusive_mask = mask[:,0:1,:,:], exclusive_mask[:,0:1,:,:]

        model.eval()
        
        out_train = model(stack)

        loss = criterion(out_train, target, flow, mask, exclusive_mask, args['zero_flow']) 
        loss.backward()

        ## Do the backward and step every loss.backward()
        if training%args['nb_trainings_before_step'] == 0 and training>=1:
            print("%04d"%training)
            optimizer.step()
            del loss
            optimizer.zero_grad()

            if training%(25*args['nb_trainings_before_step']) == 0:
                torch.save(model.state_dict(), output_path + args['final_name'])
     

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="MF2F offline no teacher")
    parser.add_argument("--input"                   , type=str  , default=""               , help='path to input frames (C type)')
    parser.add_argument("--target"                  , type=str  , default="same_as_input"  , help='path to target frames (C type). By default, they are taken in the same folder as the noisy input.')
    # Added by Diego
    ####################################
    parser.add_argument("--loss"                    , type=str  , choices=("sure", "pure", "pgure", "unsure", "unpgure", "r2r_g", "r2r_p"), required=True)
    parser.add_argument("--sigma"                   , type=float, default=None) #Gaussian std
    parser.add_argument("--gamma"                   , type=float, default=None) #Poisson scalar factor
    parser.add_argument("--step_size"               , type=float, default=1e-4, help="Gradient step size") #UNSURE and PG-UNSURE 
    parser.add_argument("--momentum"                , type=float, default=0.9, help="Gradient momentum")    #UNSURE and PG-UNSURE 
    ####################################
    parser.add_argument("--ref"                     , type=str  , default=""               , help='path to reference frames (C type), against which the psnr is going to be computed')
    parser.add_argument("--image_name"              , type=str  , default="%03d.tif"       , help='name of image file')
    parser.add_argument("--flow"                    , type=str  , default=""               , help='path to optical flow (C type)')
    parser.add_argument("--mask_collision"          , type=str  , default=""               , help='path to_collision mask(C type)')
    parser.add_argument("--mask_warping_res"        , type=str  , default=""               , help='path to_warping_res mask(C type)')
    parser.add_argument("--output"                  , type=str  , default="./%03d.png"     , help='path to output image (C type)')
    parser.add_argument("--final_name"              , type=str  , default="mf2f.pth"       , help='final name of the stored weights in the output folder')
    parser.add_argument("--range"                   , type=int  , default=255              , help='images will be divided by range (255 or 1 or 4095, etc.)')
    parser.add_argument("--crops_size"              , type=int  , default=512              , help='crops_size during training')
    parser.add_argument("--nb_trainings"            , type=int  , default=4002             , help='number of trainings')
    parser.add_argument("--nb_trainings_before_step", type=int  , default=20               , help='number of trainings before each step')
    parser.add_argument("--network"                 , type=str  , default="model/model.pth", help='path to the network')
    parser.add_argument("--lr"                      , type=float, default=0.00001          , help='learning rate')
    parser.add_argument("--zero_flow"               , action='store_true'                  , help='put this flag when flow is 0. This means no mask used, no warping, etc.')

    argspar = parser.parse_args()

    if argspar.target == 'same_as_input':
        argspar.target = argspar.input

    print("\n### MF2F offline no teacher ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    MF2F(**vars(argspar))
