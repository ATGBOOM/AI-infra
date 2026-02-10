print("starting")
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
import time
torch.backends.cudnn.benchmark = True

# Check GPU is visible
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Load tiny GPT-2 (124M params)
model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Count parameters
params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {params:,}")

# Fake data (random tokens) - we don't care about quality, just speed
batch_size = 64  # <-- WE WILL BREAK THIS
seq_len = 512

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
model.train()

print(f"\n--- Training with batch_size={batch_size}, seq_len={seq_len} ---")

for step in range(10):
    # Random input
    input_ids = torch.randint(0, 50257, (batch_size, seq_len)).cuda()
    
    # Forward pass
    start = time.time()
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    
    # Backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Memory check
    mem_used = torch.cuda.memory_allocated() / 1e9
    mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"Step {step}: loss={loss.item():.3f}, mem={mem_used:.1f}/{mem_total:.1f}GB, time={time.time()-start:.2f}s")

print("\n✓ Completed without crashing")