import torch
import deepinv
from model import FastDVDnet
from dataset import FastDVDnetDataset
from torch.utils.data import DataLoader

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_dirs = ['/mnt/bdisk/dewil/loreal_POC2/sequences_almost_Poisson']
dataset = FastDVDnetDataset(base_dirs=base_dirs, patch_size=(256, 256))
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = FastDVDnet(num_input_frames=5).to(device)
physics = deepinv.physics.Denoising(noise_model=deepinv.physics.PoissonNoise(gain=65535.0)).to(device)
loss_fn = deepinv.loss.SurePoissonLoss(gain=65535.0).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-8)

for epoch in range(15):
    model.train()
    running_loss = 0.0
    
    for i, (stack, target) in enumerate(dataloader):
        stack = stack.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        output = model(stack)
        loss = loss_fn(output, target, physics, model).mean()
        loss.backward()
        
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        
        if len(grads) > 0:
            max_grad = max(g.abs().max().item() for g in grads)
            if max_grad == 0.0:
                print(f"\n[!] GRADIENTS COLLAPSED TO EXACTLY ZERO at Epoch {epoch+1}, Batch {i}")
                print(f"Stack min/max: {stack.min().item():.6f} / {stack.max().item():.6f}")
                print(f"Output min/max: {output.min().item():.6f} / {output.max().item():.6f}")
                print(f"Loss: {loss.item():.6f}")
                
                # Check for NaNs or Infs in weights
                has_nan = any(torch.isnan(p).any().item() for p in model.parameters())
                print(f"Has NaN weights: {has_nan}")
                
                exit(1)
        
        optimizer.step()
        running_loss += loss.item()
        
    print(f"Epoch {epoch+1} done. Loss: {running_loss/len(dataloader):.6f}. Max Grad: {max_grad:.6f}")

print("Training finished without collapse.")
