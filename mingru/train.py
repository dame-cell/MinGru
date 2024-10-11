import torch 
import accelerate
import torch.nn as nn 

from datasets import load_dataset 
from accelerate.utils import tqdm 

from mingru_lm import MinGRU_LM
from utils import count_parameters

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

if __name__ == "__main__":
    model = MinGRU_LM(dim=512,num_tokens=256,num_layers=6)
    count_parameters(model)
    
    
