print("starting")
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
import time
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset
from pynvml import *
import json

BATCH_SIZE = 32
SEQ_LEN = 512
STEPS = 100


def setup_gpu():
    """Initialize GPU and print info"""
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    return handle


def load_model_and_data(batch_size=BATCH_SIZE):
    """Load GPT-2 model and tokenized dataset"""
    model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format("torch")
    
    dataloader = DataLoader(tokenized, batch_size=batch_size, shuffle=True, num_workers=0)
    
    return model, dataloader


def train_and_benchmark(model, dataloader, gpu_handle, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, num_steps=STEPS, use_mixed_precision=False):
    """Train model and collect metrics"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    
    scaler = GradScaler() if use_mixed_precision else None
    
    # Metrics accumulators
    totalTime = totalForwardTime = totalBackwardTime = totalGpuUtil = totalTokensPSec = totalMemory = 0
    
    print(f"\n--- Training with batch_size={batch_size}, seq_len={seq_len}, mixed_precision={use_mixed_precision} ---")
    
    for step, batch in enumerate(dataloader):
        start = time.time()
        
        # Data loading
        dStart = time.time()
        input_ids = batch["input_ids"].cuda()
        dTime = time.time() - dStart
        
        # Forward pass
        fStart = time.time()
        if use_mixed_precision:
            with autocast():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss
        else:
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
        torch.cuda.synchronize()
        fTime = time.time() - fStart
        
        # Backward pass
        bStart = time.time()
        if use_mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        torch.cuda.synchronize()
        bTime = time.time() - bStart
        
        # Optimizer step
        oStart = time.time()
        if use_mixed_precision:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        oTime = time.time() - oStart
        
        # Collect metrics
        mem_used = torch.cuda.memory_allocated() / 1e9
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        util = nvmlDeviceGetUtilizationRates(gpu_handle)
        gpu_util_percent = util.gpu
        
        tokens_per_step = batch_size * seq_len
        step_time = time.time() - start
        tokens_per_sec = tokens_per_step / step_time
        
        # Accumulate totals
        totalMemory += mem_used
        totalBackwardTime += bTime
        totalForwardTime += fTime
        totalGpuUtil += gpu_util_percent
        totalTime += step_time
        totalTokensPSec += tokens_per_sec
        
        if step % 10 == 0:
            print(f"Step {step}: loss={loss.item():.3f}, mem={mem_used:.1f}/{mem_total:.1f}GB, "
                  f"time={step_time:.2f}s, forward={fTime:.2f}s, backward={bTime:.2f}s, "
                  f"opt={oTime:.2f}s, data_load={dTime:.3f}s, tokens/s={tokens_per_sec:.0f}, "
                  f"gpu_util={gpu_util_percent}%")
        
        if step >= num_steps:
            break
    
    num_steps_actual = step + 1
    
    # Calculate averages
    metrics = {
        "total_steps": num_steps_actual,
        "avg_tokens_per_sec": totalTokensPSec / num_steps_actual,
        "avg_step_time_ms": (totalTime / num_steps_actual) * 1000,
        "avg_gpu_utilization": totalGpuUtil / num_steps_actual,
        "avg_memory_gb": totalMemory / num_steps_actual,
        "avg_forward_time_ms": (totalForwardTime / num_steps_actual) * 1000,
        "avg_backward_time_ms": (totalBackwardTime / num_steps_actual) * 1000,
        "backward_forward_ratio": totalBackwardTime / totalForwardTime,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9
    }
    
    return metrics


def print_summary(metrics):
    """Print benchmark summary"""
    print("\n✓ Completed without crashing")
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"Total steps: {metrics['total_steps']}")
    print(f"Avg tokens/sec: {metrics['avg_tokens_per_sec']:.0f}")
    print(f"Avg step time: {metrics['avg_step_time_ms']:.1f}ms")
    print(f"Avg GPU utilization: {metrics['avg_gpu_utilization']:.1f}%")
    print(f"Avg memory used: {metrics['avg_memory_gb']:.2f} GB")
    print(f"Avg forward time: {metrics['avg_forward_time_ms']:.1f}ms")
    print(f"Avg backward time: {metrics['avg_backward_time_ms']:.1f}ms")
    print(f"Backward/Forward ratio: {metrics['backward_forward_ratio']:.2f}x")
    print(f"Peak memory: {metrics['peak_memory_gb']:.2f} GB")
    print("="*80)


def save_metrics(metrics, config, filename):
    """Append metrics to JSON file (creates list of experiments)"""
    import os
    
    # Create new experiment entry
    experiment = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": config,
        "metrics": metrics
    }
    
    # Load existing data if file exists
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            # If it's an old single-experiment format, convert to list
            if isinstance(data, dict) and "config" in data:
                data = [data]
    else:
        data = []
    
    # Append new experiment
    data.append(experiment)
    
    # Save back
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✓ Metrics saved to {filename} (experiment #{len(data)})")


# Main execution
if __name__ == "__main__":
    # Configuration
    config = {
        "model": "gpt2",
        "batch_size": BATCH_SIZE,
        "seq_length": SEQ_LEN,
        "num_steps": STEPS,
        "mixed_precision": False
    }
    
    # Setup
    gpu_handle = setup_gpu()
    model, dataloader = load_model_and_data(batch_size=config["batch_size"])
    
    # Train and benchmark
    metrics = train_and_benchmark(
        model, 
        dataloader, 
        gpu_handle,
        batch_size=config["batch_size"],
        seq_len=config["seq_length"],
        num_steps=config["num_steps"],
        use_mixed_precision=config["mixed_precision"]
    )
    
    # Print results
    print_summary(metrics)
    
    # Save to JSON
    save_metrics(metrics, config, "benchmarks/baseline.json")