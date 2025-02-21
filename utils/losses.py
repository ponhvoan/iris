import re
import torch
import torch.nn.functional as F

class SoftCrossEntropyLoss():
    def __init__(self):
        super().__init__()

    def __call__(self, y_hat, y_soft):
        loss = -(y_soft*torch.log(y_hat)).sum()
        return loss
    

####### Sequence level metrics #######

def compute_perplexity(model, outputs, inputs):
    generated_sequence = outputs.sequences[0]

    # Determine the length of the prompt
    prompt_length = inputs['input_ids'].shape[1]
    generated_sequence = generated_sequence.unsqueeze(0).to(model.device)

    with torch.no_grad():
        outputs_full = model(generated_sequence)

    # Get logits 
    logits = outputs_full.logits  
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = generated_sequence[:, 1:].contiguous()

    # Create a mask to select the generated tokens
    gen_token_mask = torch.zeros(shift_labels.shape, dtype=torch.bool).to(model.device)
    gen_token_mask[:, prompt_length - 1:] = True  # -1 because shift_labels is offset by 1

    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    gen_token_mask = gen_token_mask.view(-1)

    gen_shift_logits = shift_logits[gen_token_mask]
    gen_shift_labels = shift_labels[gen_token_mask]
    log_probs = F.log_softmax(gen_shift_logits, dim=-1)

    # Gather the log probabilities corresponding to the actual tokens
    gen_shift_labels = gen_shift_labels.to(torch.long)
    token_log_probs = log_probs[range(gen_shift_labels.size(0)), gen_shift_labels]
    nll = -token_log_probs.sum()

    # Compute the average NLL per token
    n_tokens = gen_shift_labels.size(0)
    avg_nll = nll / n_tokens
    ppl = torch.exp(avg_nll)
    
    return ppl

def compute_entropy(model, outputs, inputs):
    generated_sequence = outputs.sequences[0] 

    prompt_length = inputs['input_ids'].shape[1]
    generated_sequence = generated_sequence.unsqueeze(0).to(model.device)  

    with torch.no_grad():
        outputs_full = model(generated_sequence)

    # Get logits
    logits = outputs_full.logits  

    # Shift logits to align tokens with their predictions
    shift_logits = logits[:, :-1, :].contiguous() 
    log_probs = F.log_softmax(shift_logits, dim=-1) 
    probs = torch.exp(log_probs)  
    entropies = - (probs * log_probs).sum(dim=-1)  

    # Create a mask to select the generated tokens
    gen_token_mask = torch.zeros(entropies.shape, dtype=torch.bool).to(model.device)
    gen_token_mask[:, prompt_length - 1:] = True  # -1 because entropies are shifted by 1

    entropies = entropies.view(-1)  # Flatten entropies and mask  
    gen_token_mask = gen_token_mask.view(-1)  
    gen_entropies = entropies[gen_token_mask]  
    avg_entropy = gen_entropies.mean()
    
    return avg_entropy

def compute_soft_labels(model, outputs, inputs):
    
    entropy = compute_entropy(model, outputs, inputs) 
    perp = compute_perplexity(model, outputs, inputs)
    return {'entropy': entropy, 'perp': perp}
 