"""
Training Script for nanoJEPA

Train the model on GSM8K with:
1. Token Loss (Autoregressive)
2. JEPA Loss (Latent Prediction)
"""

import os
import time
import math
import torch
from torch.utils.data import DataLoader

from .model import NanoJEPA
from .config import Config
from .data import GSM8KDataset, collate_fn

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
config = Config()
device = config.device
if device == 'cuda' and not torch.cuda.is_available():
    device = 'cpu'
print(f"Using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# -----------------------------------------------------------------------------
# Data Loader
# -----------------------------------------------------------------------------
train_dataset = GSM8KDataset(split='train', config=config)
# Use a smaller subset for quick educational run if needed, but GSM8K is small enough (7.5k)
# train_dataset.items = train_dataset.items[:1000] 

train_loader = DataLoader(
    train_dataset, 
    batch_size=config.batch_size, 
    shuffle=True, 
    collate_fn=collate_fn,
    drop_last=True
)

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
# Update vocab size in config if dataset added new tokens (tiktoken is fixed though)
# We assumed manual token IDs for SEP and PRED in config.
# Ensure embedding layer is big enough.
config.vocab_size = config.final_vocab_size 

model = NanoJEPA(config)
model.to(device)

# -----------------------------------------------------------------------------
# Optimizer
# -----------------------------------------------------------------------------
optimizer = model.configure_optimizers(weight_decay=1e-1, learning_rate=config.learning_rate, betas=(0.9, 0.95), device_type=device)

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
model.train()
iter_num = 0
t0 = time.time()

print("Starting training...")

whileiter = iter(train_loader)
# Epochs loop implicitly handled by recreating iterator or just looping max_iters
# For educational simplicity, we just do one pass or fixed iterations
data_iter = iter(train_loader)

for iter_num in range(config.max_iters):
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        batch = next(data_iter)
    
    # Move to device
    input_ids = batch['input_ids'].to(device)
    q_lens = batch['q_lens'].to(device)
    a_lens = batch['a_lens'].to(device)
    
    # Forward
    # We use input_ids as targets for token loss
    outputs = model(input_ids, q_lens=q_lens, a_lens=a_lens, targets=input_ids)
    
    loss = outputs['loss']
    token_loss = outputs['token_loss']
    jepa_loss = outputs['jepa_loss']
    
    # Backward
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    
    # Logging
    if iter_num % 10 == 0:
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        print(f"iter {iter_num}: loss {loss.item():.4f}, token_loss {token_loss.item():.4f}, jepa_loss {jepa_loss.item():.4f}, time {dt*1000:.2f}ms")
        
        if torch.isnan(loss):
            print("CRITICAL: Loss is NaN! Stopping training.")
            break
        
        if torch.isnan(loss):
            print("CRITICAL: Loss is NaN! Stopping training.")
            break

print("Training finished.")

# Save model
os.makedirs("out", exist_ok=True)
torch.save(model.state_dict(), "out/nanojepa.pt")
print("Model saved to out/nanojepa.pt")
