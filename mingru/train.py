import torch 
import accelerate
import torch.nn as nn 
from torch.utils.data import DataLoader 

from accelerate.utils import tqdm 

from mingru_lm import MinGRU_LM
from utils import count_parameters , decode_tokens
from pytorch_utils import  MinGruDataset
from transformers import get_linear_schedule_with_warmup


if __name__ == "__main__":
    model = MinGRU_LM(dim=512,num_tokens=256,num_layers=6)
    count_parameters(model)

    train_data  = MinGruDataset("path_to_train_data")
    test_data  = MinGruDataset("path_to_test_data")
    
    train_dataloader = DataLoader(dataset=train_data,batch_size=4,shuffle=True)
    test_dataloader = DataLoader(dataset=test_data,batch_size=4,shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    num_training_steps = len(train_dataloader) * 50
    num_warmup_steps = int(0.1 * num_training_steps)  
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)          
    

    