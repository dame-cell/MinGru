

def decode_tokens(tokens):
    return ''.join([chr(token) for token in tokens if token >= 32 and token < 256]) 


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    formatted_params = f"{total_params:,}"
    print(f"Total trainable parameters: {formatted_params}")
    return total_params

def default(v, d):
    return v if exists(v) else d
    
def exists(v):
    return v is not None