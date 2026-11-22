"""Losses written on per-token log-probabilities, so every one of them runs through the pipeline unchanged.

Each function takes torch tensors and returns a scalar torch loss. `logprobs` has shape (rows, tokens - 1): the log-probability of
every next token. `mask` marks the tokens that count (the completion), and `sequence_logprob` turns the two into one number per row."""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

PIECE_BYTES = 1 << 28


def _piece_rows(vocab: int) -> int:
    return max(1, PIECE_BYTES // (4 * vocab))


def _piece_logprobs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    x = logits.float()
    return x.gather(-1, targets[:, None]).squeeze(-1) - torch.logsumexp(x, -1)


def token_logprobs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """log p(target) for every position, shape targets.shape, without a float32 copy of the whole logits tensor: the rows go
    through in pieces of about PIECE_BYTES, and in training each piece is recomputed in the backward pass instead of kept."""
    vocab = logits.shape[-1]
    flat, t = logits.reshape(-1, vocab), targets.reshape(-1).to(logits.device)
    rows = _piece_rows(vocab)
    if flat.shape[0] <= rows:
        return _piece_logprobs(flat, t).view(targets.shape)
    keep = torch.is_grad_enabled() and flat.requires_grad
    parts = [checkpoint(_piece_logprobs, x, y, use_reentrant=False) if keep else _piece_logprobs(x, y)
             for x, y in zip(flat.split(rows), t.split(rows))]
    return torch.cat(parts).view(targets.shape)


def token_stats(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """log p(target), the entropy of each next-token distribution, and whether the target was the most likely token, piece by
    piece (no gradients)."""
    vocab = logits.shape[-1]
    flat, t = logits.reshape(-1, vocab), targets.reshape(-1).to(logits.device)
    lp, entropy, top1 = [], [], []
    with torch.no_grad():
        for x, y in zip(flat.split(_piece_rows(vocab)), t.split(_piece_rows(vocab))):
            logp = torch.log_softmax(x.float(), -1)
            lp.append(logp.gather(-1, y[:, None]).squeeze(-1))
            entropy.append(-(logp.exp() * logp).sum(-1))
            top1.append((logp.argmax(-1) == y).float())
    return tuple(torch.cat(v).view(targets.shape) for v in (lp, entropy, top1))


def lean_causal_loss(original):
    """transformers' causal LM loss computed with token_logprobs when the logits are large, so a model with a big vocabulary does
    not hold float32 copies of all of them (the loss and its gradients are the same); small logits go to `original`."""
    def loss(logits, labels, vocab_size, num_items_in_batch=None, ignore_index=-100, shift_labels=None, **kwargs):
        if logits.numel() <= 2 * _piece_rows(vocab_size) * vocab_size:
            return original(logits, labels, vocab_size, num_items_in_batch=num_items_in_batch, ignore_index=ignore_index,
                            shift_labels=shift_labels, **kwargs)
        if shift_labels is None:
            shift_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:]
        t = shift_labels.reshape(-1).to(logits.device)
        counted = t != ignore_index
        lp = token_logprobs(logits.reshape(-1, vocab_size), t.masked_fill(~counted, 0))
        total = -(lp * counted).sum()
        if num_items_in_batch is None:
            return total / counted.sum()
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(total.device)
        return total / num_items_in_batch
    return loss


def sequence_logprob(logprobs: torch.Tensor, mask: torch.Tensor, average: bool = False) -> torch.Tensor:
    total = (logprobs * mask).sum(-1)
    return total / mask.sum(-1).clamp(min=1) if average else total


def cross_entropy(logprobs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return -(logprobs * mask).sum() / mask.sum().clamp(min=1)


def dpo(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor, ref_chosen: torch.Tensor, ref_rejected: torch.Tensor, beta: float = 0.1,
        label_smoothing: float = 0.0) -> torch.Tensor:
    margin = beta * ((policy_chosen - ref_chosen) - (policy_rejected - ref_rejected))
    return (-F.logsigmoid(margin) * (1 - label_smoothing) - F.logsigmoid(-margin) * label_smoothing).mean()


def ipo(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor, ref_chosen: torch.Tensor, ref_rejected: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    margin = (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
    return ((margin - 1 / (2 * beta)) ** 2).mean()


def simpo(policy_chosen_avg: torch.Tensor, policy_rejected_avg: torch.Tensor, beta: float = 2.0, gamma: float = 0.5) -> torch.Tensor:
    return -F.logsigmoid(beta * (policy_chosen_avg - policy_rejected_avg) - gamma).mean()


def orpo(chosen_avg: torch.Tensor, rejected_avg: torch.Tensor, chosen_nll: torch.Tensor, lam: float = 0.1) -> torch.Tensor:
    log_odds = (chosen_avg - rejected_avg) - (torch.log1p(-torch.exp(chosen_avg).clamp(max=1 - 1e-6)) - torch.log1p(-torch.exp(rejected_avg).clamp(max=1 - 1e-6)))
    return chosen_nll.mean() - lam * F.logsigmoid(log_odds).mean()


def policy_gradient(logprobs: torch.Tensor, mask: torch.Tensor, advantages: torch.Tensor, old_logprobs: torch.Tensor | None = None,
                    clip: float = 0.2) -> torch.Tensor:
    if old_logprobs is None:
        return -((logprobs * mask).sum(-1) * advantages).mean()
    ratio = torch.exp(logprobs - old_logprobs)
    adv = advantages[:, None]
    per_token = -torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
    return (per_token * mask).sum() / mask.sum().clamp(min=1)
