import torch
import torch.nn.functional as F
from torch.nn import Linear, Identity, Module
from utils import exists

def heinsen_associative_scan_log(log_coeffs, log_values):
    a_star = log_coeffs.cumsum(dim=1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim=1)
    log_h = a_star + log_h0_plus_b_star
    return log_h.exp()

def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

class MinGRU(Module):
    def __init__(self, dim, expansion_factor=1.):
        super().__init__()
        dim_inner = int(dim * expansion_factor)
        # Combined transformation for hidden state and gate
        self.to_hidden = Linear(dim, dim_inner, bias=False)
        self.to_gate = Linear(dim,dim_inner,bias=False)
        # Output projection (Identity if no expansion)
        self.to_out = Linear(dim_inner, dim, bias=False) if expansion_factor != 1. else Identity()
    
    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False):
        # Split combined transformation into hidden and gate components
        hidden= self.to_hidden(x)
        gate = self.to_gate(x) 
        # Convert to log space for numerical stability
        log_coeffs = -F.softplus(gate)           # log(1 - σ(gate))
        log_z = -F.softplus(-gate)               # log(σ(gate))
        log_tilde_h = log_g(hidden)              # log(g(hidden))
        log_values = log_z + log_tilde_h         # log(z * h_tilde)
        
        # Handle previous hidden state if it exists
        if exists(prev_hidden):
            log_values = torch.cat((log_g(prev_hidden), log_values), dim=1)
            log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))
        
        # Apply parallel scan in log space
        out = heinsen_associative_scan_log(log_coeffs, log_values)
        out = out[:, -x.shape[1]:]  # Keep only the relevant sequence length
        
        # Store last hidden state for potential return
        next_prev_hidden = out[:, -1:]
        
        # Apply output projection
        out = self.to_out(out)
        
        if not return_next_prev_hidden:
            return out
        return out, next_prev_hidden

if __name__ == "__main__":
    x = torch.rand(2,256,512)
    model = MinGRU(dim=512)
    out , next_prev_hidden = model(x,return_next_prev_hidden=True)


    print("out",out[0,0,:3])
    print("next_prev_hidden",next_prev_hidden[0,0,:3])
    print("out shape",out.shape)
    print("X shape",x.shape)
    assert x.shape == out.shape 
