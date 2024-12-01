import torch

def batch_acc(pred: torch.Tensor, target: torch.Tensor, pad_token: int) -> float:
    """
    Args:
        pred: Predicted token sequence tensor.
        target: Target token sequence tensor.
    """
    pred_len = pred.shape[1]
    target_len = target.shape[1]

    if pred_len > target_len:
        pad = torch.full((target.shape[0], pred.shape[1] - target.shape[1]), pad_token).to(target.device)
        target = torch.cat((target, pad), dim=1)
    elif pred_len < target_len:
        # Pad prediction with pad_token
        pad = torch.full((pred.shape[0], target.shape[1] - pred.shape[1]), pad_token).to(pred.device)
        pred = torch.cat((pred, pad), dim=1)

    print((pred != pad_token))
    mask = torch.logical_or((pred != pad_token), (target != pad_token))
    seq_lens = mask.sum(dim=1)
    print(mask)
    print(seq_lens)
    
    # Calculate number of correct tokens for each row
    correct = ((pred == target) & mask).sum(dim=1)
    print(correct)

    # Calculate average accuracy over batch
    batch_acc = (correct.float() / seq_lens.float()).mean().item()

    return batch_acc


if __name__ == '__main__':

    pred = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]])
    target = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 6, 999, 999, 999]])

    acc = batch_acc(pred, target, eos_token=997, pad_token=999)
    print(f'Batch accuracy: {acc:.3f}')