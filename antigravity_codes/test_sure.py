import torch
import deepinv
from model import FastDVDnet

torch.manual_seed(42)

# Gain is how many photons represent a max value of 1.0. If data is 0-1, the gain should be high (e.g. 65535)
gain = 65535.0

physics = deepinv.physics.Denoising(noise_model=deepinv.physics.PoissonNoise(gain=gain))

# For deepinv, SurePoissonLoss calculates terms like y^2/(gain * output + eps)
# Let's see if we input 0-1 range images, if the loss blows up.
loss_fn = deepinv.loss.SurePoissonLoss(gain=gain)

model = FastDVDnet(num_input_frames=5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Create mock data in [0, 1] range representing normalized 16-bit uint.
stack = torch.rand(2, 5, 256, 256)
target = torch.rand(2, 1, 256, 256)

for i in range(10):
    optimizer.zero_grad()
    output = model(stack)
    
    # ensure output is somewhat positive
    # output = torch.relu(output) + 1e-6
    
    loss = loss_fn(output, target, physics, model).mean()
    loss.backward()
    
    max_grad = 0.0
    has_nan = False
    for p in model.parameters():
        if p.grad is not None:
            max_grad = max(max_grad, p.grad.abs().max().item())
            if torch.isnan(p.grad).any():
                has_nan = True
                
    print(f"Step {i}: Loss = {loss.item():.4f}, Max Grad = {max_grad:.6f}, NaN Grad = {has_nan}")
    
    # Clip grads to prevent exploding gradients blowing up weights
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
