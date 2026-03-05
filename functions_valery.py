## Modules
import iio
import numpy as np
import os
from skimage.metrics import structural_similarity
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from random import choices

##Functions 


def build_variance_map(u1, u2, u3, u4, u5, u6, u7, u8, s1, s2, s3, s4, s5, s6, s7, s8):
    variance_map = s1*torch.tensor(u1) + s2*torch.tensor(u2) + s3*torch.tensor(u3) + s4*torch.tensor(u4) + s5*torch.tensor(u5) + s6*torch.tensor(u6) + s7*torch.tensor(u7) + s8*torch.tensor(u8)
    variance_map = torch.unsqueeze(variance_map, 0)
    variance_map = torch.unsqueeze(variance_map, 0)
    variance_map = variance_map.cuda()
    return variance_map

check_string = lambda string : all(c in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] for c in string)

def create_parameter(value):
    param = Variable(torch.tensor([value], requires_grad = True).cuda())
    param.requires_grad = True
    param = torch.nn.Parameter(param)
    return param

def find_brightness(u):
    brightness = give_brightness(u)
    u1 = (brightness < 0.125 )
    u2 = (brightness >= 0.125) * (brightness < 0.25 )
    u3 = (brightness >= 0.25 ) * (brightness < 0.375)
    u4 = (brightness >= 0.375) * (brightness < 0.5  )
    u5 = (brightness >= 0.5  ) * (brightness < 0.625)
    u6 = (brightness >= 0.625) * (brightness < 0.75 )
    u7 = (brightness >= 0.75 ) * (brightness < 0.875)
    u8 = (brightness >= 0.875) 
    return(u1, u2, u3, u4, u5, u6, u7, u8)

def gives_flow(path_flow, H, W):
    flow = read_flow(path_flow)[:H, :W]
    flow = np.expand_dims(flow, 0)
    flow = Variable(torch.Tensor(flow))
    flow = flow.permute(0, 3, 1, 2)
    return flow

give_brightness = lambda x : torch.mean(x, 0) #average on the three channels

def gives_masks(path_collition, path_warping_error_based, H, W):
    if check_string(path_collition):
        mask1 = np.ones((H, W, 1))
        exclusive_mask1 = np.ones((H, W, 1))
    else:
        mask0 = iio.read(path_collition)[:H, :W]
        mask1 = 1. * (mask0 < 1.5)*(mask0 > 0.5)
        mask1 = 1. - (mask1 > 0)
        exclusive_mask1 = 1. * (mask0 >= 1.5)
        exclusive_mask1 = 1. - exclusive_mask1
    if check_string(path_warping_error_based):
        mask0 = np.ones((H, W, 1))
        exclusive_mask_warping_res = np.ones((H, W, 1))
    else:
        mask_warping_res = iio.read(path_warping_error_based)[:H, :W]
        mask0 = 1. * (mask_warping_res < 1.5)*(mask_warping_res > 0.5)
        exclusive_mask_warping_res = 1. * (mask_warping_res >= 1.5)
        exclusive_mask_warping_res = 1. - exclusive_mask_warping_res

    exclusive_mask = exclusive_mask1 * exclusive_mask_warping_res
    mask = mask1 * mask0
    
    mask = mask.transpose(2,0,1)
    exclusive_mask = exclusive_mask.transpose(2,0,1)
    mask = np.expand_dims(mask, 0)
    mask = torch.Tensor(mask).view(1,1,H,W).repeat(1,3,1,1).cuda()
    exclusive_mask = np.expand_dims(exclusive_mask, 0)
    exclusive_mask = torch.Tensor(exclusive_mask).view(1,1,H,W).repeat(1,3,1,1).cuda()

    return mask, exclusive_mask

def gives_raw_masks(path_collition, path_warping_error_based, H, W):
    if check_string(path_collition):
        mask1 = np.ones((H, W, 1))
        exclusive_mask1 = np.ones((H, W, 1))
    else:
        mask0 = iio.read(path_collition)[:H, :W]
        mask1 = 1. * (mask0 < 1.5)*(mask0 > 0.5)
        mask1 = 1. - (mask1 > 0)
        exclusive_mask1 = 1. * (mask0 >= 1.5)
        exclusive_mask1 = 1. - exclusive_mask1
    if check_string(path_warping_error_based):
        mask0 = np.ones((H, W, 1))
        exclusive_mask_warping_res = np.ones((H, W, 1))
    else:
        mask_warping_res = iio.read(path_warping_error_based)[:H, :W]
        mask0 = 1. * (mask_warping_res < 1.5)*(mask_warping_res > 0.5)
        exclusive_mask_warping_res = 1. * (mask_warping_res >= 1.5)
        exclusive_mask_warping_res = 1. - exclusive_mask_warping_res

    exclusive_mask = exclusive_mask1 * exclusive_mask_warping_res
    mask = mask1 * mask0
    
    mask = mask.transpose(2,0,1)
    exclusive_mask = exclusive_mask.transpose(2,0,1)
    mask = np.expand_dims(mask, 0)
    mask = torch.Tensor(mask).view(1,1,H,W).repeat(1,4,1,1).cuda()
    exclusive_mask = np.expand_dims(exclusive_mask, 0)
    exclusive_mask = torch.Tensor(exclusive_mask).view(1,1,H,W).repeat(1,4,1,1).cuda()

    return mask, exclusive_mask


learning_rate = lambda x : 10**-5 * np.exp(-5*x)


def linear_transform(data, a, b, u=1.4, v=0, inverse=False):
    alpha =  u/a
    beta  = (u*b/a - a*v/u) / a

    if inverse:
        return (data-beta) / alpha
    return alpha*data+beta


def normalize_augment(input_stack, target):
	'''Normalizes and augments an input patch of dim [N, num_frames, C. H, W] in [0., 255.] to \
		[N, num_frames*C. H, W] in  [0., 1.]. It also returns the central frame of the temporal \
		patch as a ground truth.
	'''
	def transform(sample1, sample2):
		# define transformations
		do_nothing = lambda x: x
		do_nothing.__name__ = 'do_nothing'
		flipud = lambda x: torch.flip(x, dims=[2])
		flipud.__name__ = 'flipup'
		rot90 = lambda x: torch.rot90(x, k=1, dims=[2, 3])
		rot90.__name__ = 'rot90'
		rot90_flipud = lambda x: torch.flip(torch.rot90(x, k=1, dims=[2, 3]), dims=[2])
		rot90_flipud.__name__ = 'rot90_flipud'
		rot180 = lambda x: torch.rot90(x, k=2, dims=[2, 3])
		rot180.__name__ = 'rot180'
		rot180_flipud = lambda x: torch.flip(torch.rot90(x, k=2, dims=[2, 3]), dims=[2])
		rot180_flipud.__name__ = 'rot180_flipud'
		rot270 = lambda x: torch.rot90(x, k=3, dims=[2, 3])
		rot270.__name__ = 'rot270'
		rot270_flipud = lambda x: torch.flip(torch.rot90(x, k=3, dims=[2, 3]), dims=[2])
		rot270_flipud.__name__ = 'rot270_flipud'
		add_csnt = lambda x: x + torch.normal(mean=torch.zeros(x.size()[0], 1, 1, 1), \
								 std=(5/255.)).expand_as(x).to(x.device)
		add_csnt.__name__ = 'add_csnt'

		# define transformations and their frequency, then pick one.
		aug_list = [do_nothing, flipud, rot90, rot90_flipud, \
					rot180, rot180_flipud, rot270, rot270_flipud, add_csnt]
		w_aug = [32, 12, 12, 12, 12, 12, 12, 12, 12] # one fourth chances to do_nothing
		transf = choices(aug_list, w_aug)

		# transform all images in array
		return transf[0](sample1), transf[0](sample2)

	#augment
	img_train, target = transform(input_stack, target)

	## extract ground truth (central frame)
	#gt_train = img_train[:, 3*ctrl_fr_idx:3*ctrl_fr_idx+3, :, :]
	#return img_train, gt_train
	return img_train, target


def psnr(img1, img2, peak=1):
    '''
    Computes the PSNR 'metric' between two images assumed to be in the range [0,1]
    '''
    x = ((np.array(img1).squeeze() - np.array(img2).squeeze()).flatten())
    return (10*np.log10(peak**2 / np.mean(x**2)))

def psnr_batch(batch1, batch2, peak=1):
    x = (batch1 - batch2).flatten()
    return (10*torch.log10(peak**2 / torch.mean(x**2))).item()

def read_flow(file):
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file)is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[:-4]
    f = open(file, 'rb')
    flo_number = np.fromfile(f, np.float32, count = 1)[0]
    assert flo_number == 202021.25, "Flow number %r incorrect. Invalid .flo file" % flo_number
    w = np.fromfile(f, np.int32, count = 1)
    h = np.fromfile(f, np.int32, count = 1)
    data = np.fromfile(f, np.float32, count = 2*w[0]*h[0])
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()
    return flow

def reads_image(path, H, W, im_range=255):
    image = iio.read(path)[:H, :W]
    image = image / im_range
    image = image.transpose(2,0,1)
    image = torch.Tensor(image).cuda()
    image = Variable(image)
    return image

def save_image(path, img):
    img = image.detach().cpu().numpy.squeeze()
    img = img.transpose(1,2,0)
    img = (img*255).astype(np.uint8)
    iio.write(path, img)

ssim = lambda x, y : structural_similarity(x.astype(float),y.astype(float),multichannel=True, data_range=255)

def temp_denoise(model, noisyframe, sigma_noise):
    '''Encapsulates call to denoising model and handles padding.
        Expects noisyframe to be normalized in [0., 1.]
    '''
    # make size a multiple of four (we have two scales in the denoiser)
    sh_im = noisyframe.size()
    expanded_h = sh_im[-2]%4
    if expanded_h:
        expanded_h = 4-expanded_h
    expanded_w = sh_im[-1]%4
    if expanded_w:
        expanded_w = 4-expanded_w
    padexp = (0, expanded_w, 0, expanded_h)
    noisyframe = F.pad(input=noisyframe, pad=padexp, mode='reflect')
    sigma_noise = F.pad(input=sigma_noise, pad=padexp, mode='reflect')

    # denoise
    out = model(noisyframe, sigma_noise)

    if expanded_h:
        out = out[:, :, :-expanded_h, :]
    if expanded_w:
        out = out[:, :, :, :-expanded_w]

    return out

def temp_denoise_8_sigmas(model, noisyframe, sigma_noise1, sigma_noise2, sigma_noise3):
    '''Encapsulates call to denoising model and handles padding.
        Expects noisyframe to be normalized in [0., 1.]
    '''
    # make size a multiple of four (we have two scales in the denoiser)
    sh_im = noisyframe.size()
    expanded_h = sh_im[-2]%4
    if expanded_h:
        expanded_h = 4-expanded_h
    expanded_w = sh_im[-1]%4
    if expanded_w:
        expanded_w = 4-expanded_w
    padexp = (0, expanded_w, 0, expanded_h)
    noisyframe = F.pad(input=noisyframe, pad=padexp, mode='reflect')
    sigma_noise1 = F.pad(input=sigma_noise1, pad=padexp, mode='reflect')
    sigma_noise2 = F.pad(input=sigma_noise2, pad=padexp, mode='reflect')
    sigma_noise3 = F.pad(input=sigma_noise3, pad=padexp, mode='reflect')

    # denoise
    out = model(noisyframe, sigma_noise1, sigma_noise2, sigma_noise3)

    if expanded_h:
        out = out[:, :, :-expanded_h, :]
    if expanded_w:
        out = out[:, :, :, :-expanded_w]

    return out

def temp_denoise_raw(model, noisyframe):
    '''Encapsulates call to denoising model and handles padding.
        Expects noisyframe to be normalized in [0., 1.]
        Par rapport à l'autre, la différence est qu'ici il n'y a pas de noise map. C'est utile pour le model raw.
    '''
    # make size a multiple of four (we have two scales in the denoiser)
    sh_im = noisyframe.size()
    expanded_h = sh_im[-2]%4
    if expanded_h:
        expanded_h = 4-expanded_h
    expanded_w = sh_im[-1]%4
    if expanded_w:
        expanded_w = 4-expanded_w
    padexp = (0, expanded_w, 0, expanded_h)
    noisyframe = F.pad(input=noisyframe, pad=padexp, mode='reflect')

    # denoise
    out = model(noisyframe)

    if expanded_h:
        out = out[:, :, :-expanded_h, :]
    if expanded_w:
        out = out[:, :, :, :-expanded_w]

    return out

def tensor_to_image(tensor):
    image = torch.clamp(tensor, 0, 1)
    image = np.array(image.cpu())
    image = np.squeeze(image)
    image = image.transpose(1,2,0)
    return image
