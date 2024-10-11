import os
import torch
import wandb
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from accelerate.utils import tqdm
import torch.multiprocessing as mp
from mingru_lm import MinGRU_LM
from utils import count_parameters, decode_tokens, tokenize_text
from pytorch_utils import MinGruDataset
from transformers import get_linear_schedule_with_warmup
import argparse 


def parse_args():
    parser = argparse.ArgumentParser(description="Processing data for the model")
    parser.add_argument('--dim', type=int, default=512, help="Dimension for the model")
    parser.add_argument('--num_tokens', type=int, default=256, help="Maximum tokens for model (max is 256 due to ASCII)")
    parser.add_argument('--num_layers', type=int, default=6, help="Number of layers to train the model")
    parser.add_argument('--path_to_train_data', type=str, required=True, help="Path to your saved train processed data")
    parser.add_argument('--path_to_test_data', type=str, required=True, help="Path to your saved test processed data")
    parser.add_argument('--batch_size', type=int, default=102, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=4e-3, help="Learning rate for training the model")
    parser.add_argument('--wd', type=float, default=1e-2, help="Weight decay for your optimizer")
    parser.add_argument('--epochs', type=int, default=40, help="Total number of epochs")
    parser.add_argument('--save_epoch', type=int, default=10, help="How many epochs to save each run")
    parser.add_argument('--world_size', type=int, default=2, help="Number of GPUs (DDP)")   

    return parser.parse_args()

 
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()


def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    # Create a progress bar for evaluation
    progress_bar = tqdm(total=len(dataloader), desc="Evaluating", position=0, leave=True)

    with torch.no_grad():
        for input_batch, target_batch in dataloader:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            loss, _ = model(inputs=input_batch, labels=target_batch)
            
            # Calculate number of tokens in batch (excluding padding)
            non_pad_mask = (target_batch != 0).float()
            num_tokens = torch.sum(non_pad_mask).item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            # Update the progress bar
            progress_bar.update(1)
    
    # Gather metrics from all GPUs
    total_loss = torch.tensor(total_loss).to(device)
    total_tokens = torch.tensor(total_tokens).to(device)
    
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
    
    avg_loss = (total_loss / total_tokens).item()
    perplexity = np.exp(avg_loss)

    # Close the progress bar after evaluation
    progress_bar.close()
    
    return avg_loss, perplexity


def generate_text(model, start_text="Once upon a time", max_length=100, temperature=0.7, device='cuda'):
    model.eval()
    if isinstance(model, DDP):
        model = model.module
    
    tokens = tokenize_text(start_text)
    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)  # Ensure long tensor
    
    generated_tokens = tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_length):
            _, logits = model(inputs=input_tensor, labels=None)
            
            last_token_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(last_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            if next_token < 256:  # Assuming 256 tokens for ASCII
                generated_tokens.append(next_token)
                input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], device=device)], dim=1)  # Append token
            
                if next_token == ord('.') and len(generated_tokens) > 30:
                    break
            else:
                break
    
    return decode_tokens(generated_tokens)


def main(rank,args):
    # Initialize distributed training
    setup(rank, args.world_size)
    

      # Set up device
    device = torch.device(f"cuda:{rank}")

    wandb.login(key="04098c64a0b88d5f4ff90335b7f75613041420c6")
    if rank == 0:
        wandb.init(project="gpt2-sample-fineweb-ddp", config=args, group=f"DDP_GPT2",)


    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    
    # Create model and move to GPU
    model = MinGRU_LM(dim=args.dim, num_tokens=args.num_tokens, num_layers=args.num_layers)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])
    
    if local_rank == 0:
        count_parameters(model)
    
    # Load datasets with distributed sampler
    train_data = MinGruDataset(args.path_to_train_data)
    test_data = MinGruDataset(args.path_to_test_data)
    
    train_sampler = DistributedSampler(train_data)
    test_sampler = DistributedSampler(test_data, shuffle=False)
    
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size // args.world_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )
    
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size//args.world_size,
        sampler=test_sampler,
        num_workers=2,
        pin_memory=True
    )
    
    if local_rank == 0:
        print(f"Number of training batches: {len(train_dataloader)}")
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    num_training_steps = len(train_dataloader) * args.epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    scaler = GradScaler()

    best_perplexity = float('inf')
    
    test_prompts = [
        "Once upon a time",
        "The little dog",
        "In the garden",
        "The magical wizard"
    ]
    
    for epoch in range(args.epochs):
        # Set epoch for sampler
        train_sampler.set_epoch(epoch)
        
        # Training phase
        model.train()
        model.to(device)
        train_loss = 0.0
        num_batches = len(train_dataloader)
        
        if local_rank == 0:
            progress_bar = tqdm(
                total=num_batches,
                desc=f"Epoch {epoch + 1}/{args.epochs}",
                position=0,
                leave=True
            )
        
        for step, (input_batch, target_batch) in enumerate(train_dataloader):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss, logits = model(inputs=input_batch, labels=target_batch)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update() 
            scheduler.step()
            

            train_loss += loss.item()
            current_loss = train_loss / (step + 1)
            
            if local_rank == 0:
                progress_bar.set_postfix({
                    'train_loss': f'{current_loss:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
                progress_bar.update(1)
            
            if rank == 0:
                    wandb.log({
                        "train_loss": loss.item(),
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        "epoch": epoch + 1,
                        "step": step + 1,
                    })

   
        
        # Synchronize before evaluation
        dist.barrier()
        
        # Evaluation phase
        eval_loss, eval_perplexity = evaluate_model(model, test_dataloader, device)
        
        if local_rank == 0:
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            print(f"Training Loss: {current_loss:.4f}")
            print(f"Evaluation Loss: {eval_loss:.4f}")
            print(f"Perplexity: {eval_perplexity:.2f}")
            
            # Save best model
            if eval_perplexity < best_perplexity:
                best_perplexity = eval_perplexity
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': eval_loss,
                    'perplexity': eval_perplexity,
                }, 'best_model.pt')
                print(f"New best model saved! Perplexity: {eval_perplexity:.2f}")
            
            # Generate text samples
            print("\nGenerating text samples:")
            for prompt in test_prompts:
                generated = generate_text(
                    model,
                    start_text=prompt,
                    max_length=50,
                    temperature=0.7,
                    device=device
                )
                print(f"Prompt: {prompt}\nGenerated: {generated}\n")
    if local_rank == 0:
        progress_bar.close()
    cleanup()


if __name__ == "__main__":
    args = parse_args()
    world_size = args.world_size
    mp.spawn(main, args=(args,), nprocs=world_size, join=True)
