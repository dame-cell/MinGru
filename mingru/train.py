import os
import torch
import wandb
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm 
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
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

    return parser.parse_args()


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

    avg_loss = (total_loss / total_tokens)
    perplexity = np.exp(avg_loss)

    # Close the progress bar after evaluation
    progress_bar.close()
    
    return avg_loss, perplexity


def generate_text(model, start_text="Once upon a time", max_length=100, temperature=0.7, device='cuda'):
    model.eval()
    
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


def main(args):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb
    wandb.init(project="mingru-single-gpu", config=args)

    # Create model and move to GPU
    model = MinGRU_LM(dim=args.dim, num_tokens=args.num_tokens, num_layers=args.num_layers)
    model = model.to(device)
    
    count_parameters(model)
    
    # Load datasets
    train_data = MinGruDataset(args.path_to_train_data)
    test_data = MinGruDataset(args.path_to_test_data)
    
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
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
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = len(train_dataloader)
        
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
            
            progress_bar.set_postfix({
                'train_loss': f'{current_loss:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
            progress_bar.update(1)

            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
                "step": step + 1,
            })
        
        progress_bar.close()
        
        # Evaluation phase
        eval_loss, eval_perplexity = evaluate_model(model, test_dataloader, device)
        
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"Training Loss: {current_loss:.4f}")
        print(f"Evaluation Loss: {eval_loss:.4f}")
        print(f"Perplexity: {eval_perplexity:.2f}")
        
        wandb.log({
                    "eval_loss": eval_loss,
                    "eval_perplexity":eval_perplexity
                    })
        
        # Save best model
        if eval_perplexity < best_perplexity:
            best_perplexity = eval_perplexity
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'eval_loss': eval_loss,
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


if __name__ == "__main__":
    args = parse_args()
    main(args)
