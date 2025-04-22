import torch
import torch.nn.functional as F
from tokenizer.tiktoken_loader import get_tokenizer


def generate_text(model, prompt, max_length=50, num_return_sequences=1, top_k=50):
    """
    Generate text sequences from a GPT model using top-k sampling.

    Args:
        model: GPT model instance (in eval mode).
        prompt (str): Initial text prompt.
        max_length (int): Total length of generated sequence including prompt.
        num_return_sequences (int): Number of sequences to generate.
        top_k (int): Number of top tokens to consider in sampling.

    Returns:
        List[str]: Generated text sequences.
    """
    device = next(model.parameters()).device
    enc = get_tokenizer()
    input_ids = enc.encode(prompt)
    # prepare batch of prompts
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
    input_tensor = input_tensor.unsqueeze(0).repeat(num_return_sequences, 1)

    model.eval()
    generated = input_tensor
    with torch.no_grad():
        for _ in range(max_length - input_tensor.shape[1]):
            logits, _ = model(generated)
            next_logits = logits[:, -1, :]
            # top-k filtering
            topk_probs, topk_indices = torch.topk(F.softmax(next_logits, dim=-1), top_k, dim=-1)
            next_token = torch.multinomial(topk_probs, num_samples=1)
            next_token = topk_indices.gather(-1, next_token)
            generated = torch.cat((generated, next_token), dim=1)

    # decode sequences
    sequences = [enc.decode(seq.tolist()) for seq in generated]
    return sequences

