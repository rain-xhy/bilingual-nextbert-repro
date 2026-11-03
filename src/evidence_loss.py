import torch
import torch.nn.functional as F


def evidence_objective(embeddings, cfg, teacher=None):
    q, front, middle, back, negative, deleted, sham = embeddings
    scores = torch.stack([(q * d).sum(-1).mean() for d in (front, middle, back)])
    negative_score = (q * negative).sum(-1).mean()
    deleted_score = (q * deleted).sum(-1).mean()
    sham_score = (q * sham).sum(-1).mean()
    tau = cfg.get("temperature", .05)
    selected = scores[:1] if cfg.get("front_only", False) else scores
    retrieval = F.softplus((negative_score - selected) / tau).mean()
    position = ((scores - scores.mean()) / tau).square().mean()
    sensitivity = F.softplus((cfg.get("margin", .05) + deleted_score - scores) / tau).mean()
    sham_consistency = ((sham_score - scores[1]) / tau).square()
    anchor = q.new_tensor(0.0) if teacher is None else (1 - (q * teacher).sum(-1)).mean()
    loss = retrieval + cfg.get("position_weight", 0) * position
    loss = loss + cfg.get("counterfactual_weight", 0) * (sensitivity + sham_consistency)
    loss = loss + cfg.get("anchor_weight", 0) * anchor
    return loss, {"retrieval": retrieval, "position": position, "sensitivity": sensitivity,
                  "sham_consistency": sham_consistency, "anchor": anchor}
