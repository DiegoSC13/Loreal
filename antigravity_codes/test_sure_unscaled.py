import torch
import deepinv
from model import FastDVDnet

torch.manual_seed(42)

# Gain is how many photons represent a max value of 1.0. If data is raw [0, 65535], 
# the poisson gain in physical terms should naturally be 1.0.
# If deepinv uses `gamma=1` for raw data, the SURE Poisson formula becomes y^2 / y.
# If the user sets `gamma=65535.0`, DeepInv might divide things out by 65535. Let's see.

gain = 65535.0
physics = deepinv.physics.Denoising(noise_model=deepinv.physics.PoissonNoise(gain=gain))

# Loss function
loss_fn = deepinv.loss.SurePoissonLoss(gain=gain)

model = FastDVDnet(num_input_frames=5)
# Very low LR to avoid explosion
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Use RAW counts ([0, 65535]), no max normalization.
stack = torch.rand(2, 5, 256, 256) * 65535.0
target = torch.rand(2, 1, 256, 256) * 65535.0

print("Testing with raw [0, 65535] counts:")
for i in range(10):
    optimizer.zero_grad()
    
    # Model outputs RAW counts
    output = model(stack)
    
    loss = loss_fn(output, target, physics, model).mean()
    loss.backward()
    
    max_grad = 0.0
    for p in model.parameters():
        if p.grad is not None:
            max_grad = max(max_grad, p.grad.abs().max().item())
                
    print(f"Step {i}: Output Max = {output.max().item():.2f}, Loss = {loss.item():.4f}, Max Grad = {max_grad:.6f}")
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
