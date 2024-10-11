import numpy as np
import torch
from tqdm import tqdm
import argparse
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Processing data for the model")
    parser.add_argument('--data_name', type=str, default="roneneldan/TinyStories", help="Data to be used in processing and training")
    parser.add_argument('--max_length', type=int, default=512, help="Maximum sequence length for each tokenized input")
    parser.add_argument('--train_path', type=str, default="train.npz", help="Path to save the train processed data")
    parser.add_argument('--test_path', type=str, default="test.npz", help="Path to save the test processed data")

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
    tokenized_dataset = tokenize_dataset(ds)
    padded_dataset = pad_and_batch(tokenized_dataset, max_length=args.max_length)
    inputs = padded_dataset
    
    targets = torch.roll(inputs, shifts=-1, dims=1)  
    inputs_np = inputs.cpu().numpy()  
    targets_np = targets.cpu().numpy()  


    
    np.savez(args.train_path, inputs=inputs_np)
    np.savez(args.test_path, inputs=targets_np)

    print(f"Data saved to {args.train_path},{args.test_path}")
