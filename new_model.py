import torch
import torch.nn as nn
import torch.nn.functional as F

kernel_size=4

def crop_like(src, tgt):
    _, _, h, w = tgt.shape
    src_h, src_w = src.shape[2], src.shape[3]
    h = min(h, src_h)
    w = min(w, src_w)
    return src[:, :, :h, :w].contiguous()

# No quiero usar match_size. Esto debería hacerse (y a partir de ahora se va a hacer) en dataset.py
# def match_size(src, tgt):
#     return F.interpolate(src, size=tgt.shape[2:], mode='bilinear', align_corners=False)

def debug_gradients(tensors_dict):
    """
    tensors_dict: diccionario de nombre de tensor -> tensor
    Imprime para cada tensor:
        - tipo
        - requires_grad
        - grad_fn
        - shape
    """
    print("=== DEBUG GRADIENTS ===")
    for name, tensor in tensors_dict.items():
        if not isinstance(tensor, torch.Tensor):
            print(f"{name}: NOT A TENSOR! type={type(tensor)}")
            continue
        print(f"{name}: shape={tensor.shape}, requires_grad={tensor.requires_grad}, grad_fn={tensor.grad_fn}")
    print("========================\n")

class CvBlock(nn.Module):
	'''(Conv2d => LeakyReLU) x 2'''
	def __init__(self, in_ch, out_ch):
		super(CvBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=1),
			nn.LeakyReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=1),
			nn.LeakyReLU(inplace=True)
		)

	def forward(self, x):
		return self.convblock(x)

class InputCvBlock(nn.Module):
	'''(Conv with num_in_frames groups => LeakyReLU) + (Conv => LeakyReLU)'''
	def __init__(self, num_in_frames, out_ch):
		super(InputCvBlock, self).__init__()
		self.interm_ch = 30
		self.convblock = nn.Sequential(
			nn.Conv2d(num_in_frames, num_in_frames*self.interm_ch, \
					  kernel_size=kernel_size, padding=1, groups=num_in_frames),
			nn.LeakyReLU(inplace=True),
			nn.Conv2d(num_in_frames*self.interm_ch, out_ch, kernel_size=kernel_size, padding=2),
			nn.LeakyReLU(inplace=True)
		)

	def forward(self, x):
		return self.convblock(x)

class DownBlock(nn.Module):
	'''Downscale + (Conv2d => LeakyReLU)*2'''
	def __init__(self, in_ch, out_ch):
		super(DownBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=3, stride=2),
			nn.LeakyReLU(inplace=True),
			CvBlock(out_ch, out_ch)
		)

	def forward(self, x):
		return self.convblock(x)

class UpBlock(nn.Module):
	'''(Conv2d => LeakyReLU)*2 + Upscale'''
	def __init__(self, in_ch, out_ch):
		super(UpBlock, self).__init__()
		self.convblock = nn.Sequential(
			CvBlock(in_ch, in_ch),
			nn.Conv2d(in_ch, out_ch*4, kernel_size=kernel_size, padding=3),
            nn.PixelShuffle(2)
            )

	def forward(self, x):
		return self.convblock(x)[:,:,1:-1, 1:-1]


class OutputCvBlock(nn.Module):
	'''Conv2d => LeakyReLU => Conv2d'''
	def __init__(self, in_ch, out_ch):
		super(OutputCvBlock, self).__init__()
		self.convblock = nn.Sequential(
			nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=1),
			nn.LeakyReLU(inplace=True),
			nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=2)
		)

	def forward(self, x):
		return self.convblock(x)

class DenBlock(nn.Module):

	def __init__(self, num_input_frames=3):
		super(DenBlock, self).__init__()
		self.chs_lyr0 = 32
		self.chs_lyr1 = 64
		self.chs_lyr2 = 128

		self.inc = InputCvBlock(num_in_frames=num_input_frames, out_ch=self.chs_lyr0)
		self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
		self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
		self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
		self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
		self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=1)

		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity='leaky_relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, in0, in1, in2):
		# in0, in1, in2 = x[:,0:1,...], x[:,1:2,...], x[:,2:3,...]
		x0 = self.inc(torch.cat((in0, in1, in2), dim=1)) #Check Valery's model, I change something here
		x1 = self.downc0(x0)
		x2 = self.downc1(x1)
		x2 = self.upc2(x2)
		x1 = self.upc1(x1 + x2)
		x0 = self.outc(x0 + x1)
		x = torch.clamp(in1 - x0, min=0.0)
		#x = in1 - x0
		return x


class FastDVDnet_(nn.Module):
	""" Definition of the FastDVDnet model.
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=1 grayscale)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, num_input_frames=5):
		super(FastDVDnet_, self).__init__()
		self.num_input_frames = num_input_frames
		# Define models of each denoising stage
		self.temp1 = DenBlock(num_input_frames=3)
		self.temp2 = DenBlock(num_input_frames=3)
		# Init weights
		self.reset_params()

	@staticmethod
	def weight_init(m):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity='leaky_relu')

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, x, *args, **kwargs):
		'''Args:
			x: Tensor, [N, num_frames*1, H, W] in the [0., 1.] range
		'''
		# N, C, H, W = x.shape
		# if C < self.num_input_frames:
		# 	# No hay suficientes frames, devolver el frame central o el input sin procesar
		# 	# Aquí simplemente devolvemos el frame central (o el único)
		# 	mid = C // 2
		# 	return x[:, mid:mid+1, :, :]

		# Unpack inputs
		x0, x1, x2, x3, x4 = tuple(x[:, m:m+1, :, :] for m in range(self.num_input_frames))
		# Preparar los tensores concatenados y mostrar shapes antes de enviarlos a temp1
		# First stage
		x20 = self.temp1(x0, x1, x2)
		x21 = self.temp1(x1, x2, x3)
		x22 = self.temp1(x2, x3, x4)

		#Second stage
		x = self.temp2(x20, x21, x22)

		return x
		# x0_cat = torch.cat((x0, x1, x2), dim=1)
		# x1_cat = torch.cat((x1, x2, x3), dim=1)
		# x2_cat = torch.cat((x2, x3, x4), dim=1)

		# # First stage
		# x20 = self.temp1(x0_cat)
		# x21 = self.temp1(x1_cat)
		# x22 = self.temp1(x2_cat)

		# x_cat_stage2 = torch.cat((
		# 	x20 + x1,
		# 	x21 + x2,
		# 	x22 + x3
		# ), dim=1)
		# x = self.temp2(x_cat_stage2)

		# return x


class SureWrapper(nn.Module):
    """
    Adapta FastDVDnet para ser compatible con SurePoissonLoss de DeepInverse.

    El problema: SurePoissonLoss asume que el modelo recibe y devuelve tensores
    del mismo shape (el espacio de medición). FastDVDnet recibe 5 frames [B,5,H,W]
    pero devuelve 1 frame [B,1,H,W], lo que rompe la fórmula SURE internamente.

    La solución: presentar a la loss un modelo que recibe y devuelve [B,1,H,W].
    Internamente, el wrapper guarda los 5 frames como contexto, y cuando la loss
    perturba el frame central, lo sustituye en el stack antes de llamar a FastDVDnet.

    Uso típico en el bucle de entrenamiento:
        wrapper.set_context(stack)          # stack: [B, 5, H, W]
        y_central = stack[:, 2:3, :, :]    # frame central: [B, 1, H, W]
        output = wrapper(y_central)         # equivalente a model(stack)
        loss = loss_fn(y_central, output, physics, wrapper)
    """
    def __init__(self, model: FastDVDnet_):
        super().__init__()
        self.model = model
        self._context = None  # [B, 5, H, W], se actualiza cada batch

    def set_context(self, stack: torch.Tensor):
        """Guarda el stack de 5 frames. Llamar antes de cada forward."""
        self._context = stack.detach()

    def forward(self, y_central, *args, **kwargs):
        """
        y_central: [B, 1, H, W] — el frame central (posiblemente perturbado por SURE).
        Reconstruye el stack de 5 frames sustituyendo el frame central y llama a FastDVDnet.
        """
        if self._context is None:
            raise RuntimeError("Llama a wrapper.set_context(stack) antes del forward.")
        # Clonamos para no modificar el contexto original
        stack = self._context.clone()
        # Sustituimos solo el frame central (posición 2) con el y perturbado por SURE
        stack[:, 2:3, :, :] = y_central
        return self.model(stack)