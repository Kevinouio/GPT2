import torch
import torch.nn.functional as F

def get_most_likely_row(tokens, mask, logits):
    """
    Given HellaSwag tokenized inputs and model logits, return the index of the most likely completion.

    Args:
        tokens (torch.Tensor): Shape (4, T) batch of token sequences.
        mask (torch.Tensor): Shape (4, T) binary mask where 1 indicates completion tokens.
        logits (torch.Tensor): Shape (4, T, V) model output logits.

    Returns:
        int: Index of the completion (0-3) with lowest average loss.
    """
    # shift logits and tokens for next-token loss
    shift_logits = logits[..., :-1, :].contiguous()
    shift_tokens = tokens[..., 1:].contiguous()
    # compute per-token loss
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_tokens = shift_tokens.view(-1)
    losses = F.cross_entropy(flat_logits, flat_tokens, reduction='none')
    losses = losses.view(tokens.size(0), -1)
    # apply mask and average
    mask_shift = mask[..., 1:].contiguous()
    masked = losses * mask_shift
    avg_loss = masked.sum(dim=1) / mask_shift.sum(dim=1)
    return int(avg_loss.argmin().item())
