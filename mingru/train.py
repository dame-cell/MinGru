import torch 
import accelerate
import torch.nn as nn 
from torch.utils.data import DataLoader 


from datasets import load_dataset 
from accelerate.utils import tqdm 

from mingru_lm import MinGRU_LM
from utils import count_parameters , decode_tokens
from pytorch_utils import  MinGruDataset


def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

if __name__ == "__main__":
    model = MinGRU_LM(dim=512,num_tokens=256,num_layers=6)
    count_parameters(model)

    train_data  = MinGruDataset("path_to_train_data")
    test_data  = MinGruDataset("path_to_test_data")
    
    train_dataloader = DataLoader(dataset=train_data,batch_size=4,shuffle=True)
    test_dataloader = DataLoader(dataset=test_data,batch_size=4,shuffle=True)

