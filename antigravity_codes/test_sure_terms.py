import torch
import deepinv
from model import FastDVDnet

torch.manual_seed(42)

model = FastDVDnet(num_input_frames=5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Normalized images
stack = torch.rand(2, 5, 256, 256)
target = torch.rand(2, 1, 256, 256)

gamma_param = 1.0 / 65535.0

physics = deepinv.physics.Denoising(noise_model=deepinv.physics.PoissonNoise(gain=gamma_param))
loss_fn = deepinv.loss.SurePoissonLoss(gain=gamma_param)

print("Testing with gain = 1/65535 on [0, 1] data")
for i in range(10):
    optimizer.zero_grad()
    output = model(stack)
    loss = loss_fn(output, target, physics, model).mean()
    loss.backward()
    
    max_grad = 0.0
    for p in model.parameters():
        if p.grad is not None:
            max_grad = max(max_grad, p.grad.abs().max().item())
            
    print(f"Step {i}: Loss = {loss.item():.6f}, Max Grad = {max_grad:.6f}")
    optimizer.step()
