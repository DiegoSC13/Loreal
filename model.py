import torch
import torch.nn as nn

kernel_size=4

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
					  kernel_size=kernel_size, padding=1),
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

	def forward(self, x):
		'''Args:
			inX: Tensor, [N, C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Input convolution block (with Diego's modifications)
		in0, in1, in2 = x[:,0:1,...], x[:,1:2,...], x[:,2:3,...]
		x0 = self.inc(torch.cat((in0, in1, in2), dim=1))
		# Downsampling
		x1 = self.downc0(x0)
		x2 = self.downc1(x1)
		# Upsampling
		x2 = self.upc2(x2)
		x1 = self.upc1(x1+x2)
		# Estimation
		x = self.outc(x0+x1)

		# Residual
		x = in1 - x

		return x

class FastDVDnet(nn.Module):
	""" Definition of the FastDVDnet model.
	Inputs of forward():
		xn: input frames of dim [N, C, H, W], (C=1 grayscale)
		noise_map: array with noise map of dim [N, 1, H, W]
	"""

	def __init__(self, num_input_frames=5):
		super(FastDVDnet, self).__init__()
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
		N, C, H, W = x.shape
		if C < self.num_input_frames:
			# No hay suficientes frames, devolver el frame central o el input sin procesar
			# Aquí simplemente devolvemos el frame central (o el único)
			mid = C // 2
			return x[:, mid:mid+1, :, :]

		# Unpack inputs
		x0, x1, x2, x3, x4 = tuple(x[:, m:m+1, :, :] for m in range(self.num_input_frames))
		# print('Shape de x0 (Unpacking):', x0.shape)
		# print('Shape de x1 (Unpacking):', x1.shape)
		# print('Shape de x2 (Unpacking):', x2.shape)
		# print('Shape de x3 (Unpacking):', x3.shape)
		# print('Shape de x4 (Unpacking):', x4.shape)

		# Preparar los tensores concatenados y mostrar shapes antes de enviarlos a temp1
		x0_cat = torch.cat((x0, x1, x2), dim=1)
		x1_cat = torch.cat((x1, x2, x3), dim=1)
		x2_cat = torch.cat((x2, x3, x4), dim=1)

		# print('Shape de x0_cat (antes de temp1):', x0_cat.shape)
		# print('Shape de x1_cat (antes de temp1):', x1_cat.shape)
		# print('Shape de x2_cat (antes de temp1):', x2_cat.shape)

		# First stage
		x20 = self.temp1(x0_cat)
		x21 = self.temp1(x1_cat)
		x22 = self.temp1(x2_cat)

		# Mostrar shapes después del primer stage
		# print('Shape de x20 (después de temp1):', x20.shape)
		# print('Shape de x21 (después de temp1):', x21.shape)
		# print('Shape de x22 (después de temp1):', x22.shape)

		# Segundo stage
		x_cat_stage2 = torch.cat((x20, x21, x22), dim=1)
		# print('Shape concatenada para temp2:', x_cat_stage2.shape)
		x = self.temp2(x_cat_stage2)

		# print('Shape final de x (después de temp2):', x.shape)

		return x