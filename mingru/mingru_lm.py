import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from mingru import MinGRU
from utils import exists,default ,count_parameters

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, self.dim_inner),
            nn.GELU(),  
            nn.Linear(self.dim_inner, dim)
        )

    def forward(self, x):
        return self.net(x)

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * (self.gamma + 1)

class MinGRU_Layers(nn.Module):
    def __init__(self, dim, num_tokens):
        super().__init__()
        self.emb = nn.Embedding(num_tokens, dim)
        self.rms_norm = RMSNorm(dim)
        self.gru = MinGRU(dim)
        self.ff = FeedForward(dim)
        
        self.norm = RMSNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)

    def forward(self, inputs, labels=None, is_first_layer=True, prev_hiddens=None):
        if is_first_layer:
            x = self.emb(inputs)
        else:
            x = self.emb(inputs.argmax(dim=-1))
        
        if exists(prev_hiddens):
            x = x[:, -1:]

        next_prev_hiddens = []
        prev_hiddens = iter(default(prev_hiddens, []))

        x = self.rms_norm(x)
        prev_hidden = next(prev_hiddens, None)

        min_gru_out, next_hidden = self.gru(x, prev_hidden, return_next_prev_hidden=True)

        x = min_gru_out + x
        next_prev_hiddens.append(next_hidden)
        x = self.ff(x) + x
        logits = self.to_logits(self.norm(x))

        if labels is not None:
            loss = F.cross_entropy(logits.transpose(1, 2), labels)
        else:
            loss = None

        return loss, logits, next_prev_hiddens

class MinGRU_LM(nn.Module):
    def __init__(self, dim, num_tokens, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([MinGRU_Layers(dim, num_tokens) for _ in range(num_layers)])

    def forward(self, inputs, labels):
        total_loss = 0
        hidden_states = [None] * len(self.layers)
        current_input = inputs

        for i, layer in enumerate(self.layers):
            loss, logits, next_hiddens = layer(
                inputs=current_input,
                labels=labels,
                is_first_layer=(i == 0),
                prev_hiddens=hidden_states[i]
            )
            
            if loss is not None:
                total_loss += loss
                
            current_input = logits  # Use the logits as input for the next layer
            hidden_states[i] = next_hiddens

        return total_loss / len(self.layers), logits

if __name__ == "__main__":
    dim = 512  
    num_tokens = 256  
    num_layers = 6  
    batch_size = 4  
    seq_length = 512 

    model = MinGRU_LM(dim, num_tokens, num_layers)
    count_parameters(model)

    batch_inputs = torch.randint(0, 256, (batch_size, seq_length))
    batch_labels = torch.randint(0, 256, (batch_size, seq_length))

    loss, logits = model(batch_inputs, batch_labels)
    print("Loss",loss)
    print("logits",logits[0,0,:3])