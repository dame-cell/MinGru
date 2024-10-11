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

    def forward(self, x, is_first_layer=True, prev_hiddens=None):
        if is_first_layer:
            inputs, labels = x[:, :-1], x[:, 1:]
            x = self.emb(inputs)
        else:
            labels = x.argmax(dim=-1)
            x = self.emb(labels)  

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
        loss = F.cross_entropy(logits.transpose(1, 2), labels)

        return loss, logits, next_prev_hiddens

class MinGRU_LM(nn.Module):
    def __init__(self, dim, num_tokens, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([MinGRU_Layers(dim, num_tokens) for _ in range(num_layers)])

    def forward(self, x):
        total_loss = 0
        hidden_states = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            loss, logits, next_hiddens = layer(x, is_first_layer=(i == 0), prev_hiddens=hidden_states[i])
            total_loss += loss
            x = logits  # Use the logits as input for the next layer
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
    inputs = torch.randint(0, num_tokens, (batch_size, seq_length))

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()

        loss, logits = model(inputs)
        
        loss.backward()
        optimizer.step()

        # Print epoch and loss
        print(f"Epoch [{epoch+1}/{50}], Loss: {loss.item():.4f}")

    print("Training completed!")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_loss, final_logits = model(inputs)
    
    print(f"Final Loss: {final_loss.item():.4f}")
