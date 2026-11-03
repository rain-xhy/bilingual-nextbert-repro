"""Two-pass embedding gradient cache with exact dropout RNG replay.

Implements the established gradient-cache technique (Gao et al., 2021), not a new algorithm.
Encoder forward must be free of mutable training buffers (e.g. BatchNorm running statistics).
"""
import torch


def cached_backward(forward, batches, objective, scale=1.0):
    device = batches[0]["input_ids"].device
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    embeddings, states = [], []
    with torch.no_grad():
        for batch in batches:
            state = (torch.get_rng_state(), torch.cuda.get_rng_state(device) if cuda_devices else None)
            states.append(state)
            embeddings.append(forward(batch).detach().float())
    leaves = [e.requires_grad_() for e in embeddings]
    loss, details = objective(leaves)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite cached loss")
    (loss * scale).backward()
    for batch, leaf, state in zip(batches, leaves, states):
        if leaf.grad is None:
            # Still replay this view to keep matched-compute ablations comparable.
            gradient = torch.zeros_like(leaf)
        else:
            gradient = leaf.grad
        with torch.random.fork_rng(devices=cuda_devices):
            torch.set_rng_state(state[0])
            if cuda_devices:
                torch.cuda.set_rng_state(state[1], device)
            output = forward(batch)
            (output.float() * gradient).sum().backward()
    return {"loss": float(loss.detach()), **{k: float(v.detach()) for k, v in details.items()}}
