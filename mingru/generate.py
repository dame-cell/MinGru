from utils import decode_tokens, tokenize_text
import torch 

def generate_text(model, start_text="Once upon a time", max_length=200, temperature=0.7, device='cuda'):
    model.eval()
    
    tokens = tokenize_text(start_text)
    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)  # Ensure long tensor
    
    generated_tokens = tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_length):
            _,logits = model(input_tensor,labels=None)
            
            
            last_token_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(last_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Only append if it's within the 256-character ASCII range
            if next_token < 256:  
                generated_tokens.append(next_token)
                input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], device=device)], dim=1)
            else:
                # Handle tokens outside the ASCII range (optional)
                continue  # Optionally skip or handle as needed
        
            # Optionally stop generation after 30 tokens if a period is predicted, otherwise continue
            if len(generated_tokens) >= max_length and next_token == ord('.'):
                #break

    return decode_tokens(generated_tokens)

