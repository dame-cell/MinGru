import numpy as np
import torch
from tqdm import tqdm
import argparse
from datasets import load_dataset
from utils import decode_tokens 

def parse_args():
    parser = argparse.ArgumentParser(description="Processing data for the model")
    parser.add_argument('--data_name', type=str, default="roneneldan/TinyStories", help="Data to be used in processing and training")
    parser.add_argument('--max_length', type=int, default=512, help="Maximum sequence length for each tokenized input")
    parser.add_argument('--train_path', type=str, default="train.npz", help="Path to save the train processed data")
    parser.add_argument('--test_path', type=str, default="test.npz", help="Path to save the test processed data")
    parser.add_argument('--train_size', type=int, default=900, help="Number of samples for training")
    parser.add_argument('--test_size', type=int, default=100, help="Number of samples for testing")
    return parser.parse_args()

def tokenize_text(text):
    return [ord(char) for char in text if ord(char) < 256]

def tokenize_dataset(dataset):
    tokenized_data = []
    for item in tqdm(dataset, desc="Tokenizing Dataset"):
        tokenized_text = tokenize_text(item['text'])  
        tokenized_data.append(tokenized_text)
    return tokenized_data

def pad_and_batch(tokenized_data, max_length):
    tokenized_tensors = [torch.tensor(seq[:max_length]) for seq in tokenized_data]  
    padded_data = torch.nn.utils.rnn.pad_sequence(tokenized_tensors, batch_first=True, padding_value=0)
    return padded_data

if __name__ == "__main__":
    args = parse_args()

    ds = load_dataset(args.data_name, split='train')
    ds = ds.select(range(args.train_size + args.test_size))  
    
    train_ds = ds.select(range(args.train_size))
    test_ds = ds.select(range(args.train_size, args.train_size + args.test_size))
    
    # Process train data
    print("Processing training data...")
    train_tokenized = tokenize_dataset(train_ds)
    train_padded = pad_and_batch(train_tokenized, max_length=args.max_length)
    train_targets = torch.roll(train_padded, shifts=-1, dims=1)
    
    print("Processing test data...")
    test_tokenized = tokenize_dataset(test_ds)
    test_padded = pad_and_batch(test_tokenized, max_length=args.max_length)
    test_targets = torch.roll(test_padded, shifts=-1, dims=1)
    
    np.savez(args.train_path, 
             inputs=train_padded.cpu().numpy(),
             targets=train_targets.cpu().numpy())
    
    np.savez(args.test_path,
             inputs=test_padded.cpu().numpy(),
             targets=test_targets.cpu().numpy())
    
    print(f"Data saved to {args.train_path} and {args.test_path}")
    
    # Print sample to verify
    print("\nSample from training set:")
    print("Input:", train_padded[0][:10])  
    print("Target:", train_targets[0][:10]) 
    
    print("\nSample from test set:")
    print("Input:", test_padded[0][:10])  
    print("Target:", test_targets[0][:10])  