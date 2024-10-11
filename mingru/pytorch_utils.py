import torch 
from torch.utils.data import Dataset,DataLoader
import numpy as np 




class MinGruDataset(Dataset):
    def __init__(self, npz_file_path):
        data = np.load(npz_file_path)
        self.inputs = data['inputs']
        self.targets = data['targets']
        
        self.num_samples = len(self.inputs)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_chunk = self.inputs[idx]
        target_chunk = self.targets[idx]
        return torch.tensor(input_chunk), torch.tensor(target_chunk)


train_data  = MinGruDataset(npz_file_path="/teamspace/studios/this_studio/MinGru/mingru/train.npz")
test_data  = MinGruDataset(npz_file_path="/teamspace/studios/this_studio/MinGru/mingru/test.npz")

train_dataloader = DataLoader(dataset=train_data,batch_size=4,shuffle=True)
test_dataloader = DataLoader(dataset=test_data,batch_size=4,shuffle=True)

print("Train Batch:")
for train_batch,test_batch in train_dataloader:
    print(train_batch)
    print(test_batch)
    break  

print("\nTest Batch:")
for train_batch,test_batch in test_dataloader:
    print(train_batch)
    print(test_batch)
    break  