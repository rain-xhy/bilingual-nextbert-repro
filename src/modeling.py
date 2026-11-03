"""Shared mmBERT backbone with independent retrieval and reranking branches."""
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from src.architecture import DocumentBridge, architecture_id


def pool(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return F.normalize((hidden * mask).sum(1) / mask.sum(1).clamp_min(1), dim=-1)


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg.get("revision"))
        factory = AutoModelForMaskedLM if cfg["task"] == "mlm" else AutoModel
        kwargs = {"revision": cfg.get("revision"), "attn_implementation": cfg.get("attention", "sdpa")}
        if cfg.get("attention") == "flash_attention_2":
            kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if cfg["task"] == "mlm":
            kwargs["sparse_prediction"] = True
        architecture = AutoConfig.from_pretrained(cfg["model"], revision=cfg.get("revision"))
        for key, value in cfg.get("architecture_overrides", {}).items():
            if key not in {"local_attention", "global_attn_every_n_layers"}:
                raise ValueError(f"Unvalidated architectural ablation: {key}")
            if not isinstance(value, int) or value <= 0:
                raise ValueError("Attention window and global frequency must be positive integers")
            setattr(architecture, key, value)
        if cfg["task"] == "mlm":
            architecture.sparse_prediction = kwargs.pop("sparse_prediction")
        kwargs["config"] = architecture
        base = factory.from_pretrained(cfg["model"], **kwargs)
        if base.config.model_type != "modernbert":
            raise ValueError("Training adapter targets are verified for ModernBERT/mmBERT only")
        base.config.reference_compile = False
        self.backbone = get_peft_model(base, LoraConfig(
            r=cfg.get("lora_rank", 8), lora_alpha=cfg.get("lora_rank", 8) * 2,
            lora_dropout=0.05, target_modules=["Wqkv", "Wo", "Wi"], bias="none"))
        if cfg.get("gradient_checkpointing", True):
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        bridge_cfg = cfg.get("document_bridge", {})
        # Keep bridge/head initialization independent of LoRA loading and backbone shape.
        with torch.random.fork_rng():
            torch.manual_seed(cfg.get("seed", 42) + 1000)
            self.bridge = DocumentBridge(base.config.hidden_size, **bridge_cfg) if bridge_cfg else None
            labels = cfg.get("num_labels", 3 if cfg["task"] == "classification" else 7)
            self.head = nn.Linear(base.config.hidden_size, 1 if cfg["task"] == "rerank" else labels) \
                if cfg["task"] in {"rerank", "classification", "ner"} else None

    def tokenize(self, texts, length, pairs=None):
        if pairs is not None:
            # Reserve room for the document; a very long query must not make only_second fail.
            budget = min(self.cfg.get("query_length", 128), length // 2)
            texts = [self.tokenizer.decode(self.tokenizer.encode(t, add_special_tokens=False)[:budget]) for t in texts]
        result = self.tokenizer(texts, text_pair=pairs, padding=True,
                                truncation="only_second" if pairs is not None else True,
                                max_length=length, return_tensors="pt")
        result.pop("token_type_ids", None)
        return {k: v.to(next(self.parameters()).device) for k, v in result.items()}

    def hidden(self, batch):
        inputs = {k: v for k, v in batch.items() if k != "labels"}
        base = self.backbone.get_base_model()
        encoder = base.model if self.cfg["task"] == "mlm" else self.backbone
        hidden = encoder(**inputs).last_hidden_state
        return self.bridge(hidden, inputs["attention_mask"]) if self.bridge is not None else hidden

    def embed(self, batch):
        return pool(self.hidden(batch), batch["attention_mask"])

    def logits(self, batch):
        h = self.hidden(batch)
        if self.cfg["task"] == "ner":
            return self.head(h)
        mask = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
        return self.head((h * mask).sum(1) / mask.sum(1).clamp_min(1))

    def score(self, batch):
        return self.logits(batch).squeeze(-1)

    def mlm_loss(self, batch):
        from torch.utils.checkpoint import checkpoint
        hidden = self.hidden(batch)
        selected = batch["labels"] != -100
        h, targets = hidden[selected], batch["labels"][selected]
        if not len(targets):
            raise ValueError("MLM requires masked targets")
        base = self.backbone.get_base_model()
        def loss_fn(features, labels):
            return F.cross_entropy(base.decoder(base.head(features)).float(), labels, reduction="sum")
        losses = []
        for start in range(0, len(targets), self.cfg.get("mlm_logit_chunk", 64)):
            end = start + self.cfg.get("mlm_logit_chunk", 64)
            if torch.is_grad_enabled():
                losses.append(checkpoint(loss_fn, h[start:end], targets[start:end], use_reentrant=False))
            else:
                losses.append(loss_fn(h[start:end], targets[start:end]))
        return torch.stack(losses).sum() / len(targets)

    def load_transfer(self, checkpoint, load_head=False):
        """Transfer a common adapted backbone to a task branch; never silently drop bridge weights."""
        import json
        from pathlib import Path
        from peft import set_peft_model_state_dict
        from peft.utils.save_and_load import load_peft_weights
        root = Path(checkpoint)
        source = json.loads((root / "config.json").read_text(encoding="utf-8"))
        if any(source.get(k) != self.cfg.get(k) for k in ("model", "revision", "lora_rank")) \
                or architecture_id(source) != architecture_id(self.cfg):
            raise ValueError("Checkpoint model or architecture mismatch")
        weights = load_peft_weights(str(root / "adapter"), device="cpu")
        if source["task"] == "mlm" and self.cfg["task"] != "mlm":
            weights = {k.replace("base_model.model.model.", "base_model.model."): v for k, v in weights.items()}
        if source["task"] != "mlm" and self.cfg["task"] == "mlm":
            weights = {k.replace("base_model.model.", "base_model.model.model.", 1): v for k, v in weights.items()}
        outcome = set_peft_model_state_dict(self.backbone, weights)
        if any("lora_" in k for k in outcome.missing_keys) or outcome.unexpected_keys:
            raise ValueError("Adapter state mismatch")
        if self.bridge is not None:
            self.bridge.load_state_dict(torch.load(root / "architecture.pt", map_location="cpu", weights_only=True))
        if load_head and self.head is not None:
            if source["task"] != self.cfg["task"]:
                raise ValueError("Cannot transfer task-specific heads across tasks")
            self.head.load_state_dict(torch.load(root / "head.pt", map_location="cpu", weights_only=True))

    def save(self, destination):
        from pathlib import Path
        from src.common import write_json
        path = Path(destination)
        self.backbone.save_pretrained(path / "adapter")
        self.tokenizer.save_pretrained(path / "tokenizer")
        write_json(path / "config.json", self.cfg)
        if self.bridge is not None:
            torch.save(self.bridge.state_dict(), path / "architecture.pt")
        if self.head is not None:
            torch.save(self.head.state_dict(), path / "head.pt")


def mask_tokens(batch, tokenizer, mode, probability, generator):
    """Exact per-sequence mask budget, 80/10/10 corruption; span masking is not full SpanBERT."""
    device = batch["input_ids"].device
    # Corrupt on CPU in one transfer instead of synchronizing the GPU for every token.
    inputs = batch["input_ids"].detach().cpu().clone()
    active = batch["attention_mask"].detach().cpu()
    labels = torch.full_like(inputs, -100)
    for row in range(inputs.shape[0]):
        eligible = [i for i, token in enumerate(inputs[row].tolist())
                    if active[row, i] and token not in tokenizer.all_special_ids]
        if not eligible:
            continue
        count = max(1, round(len(eligible) * probability))
        remaining = set(eligible)
        selected = []
        while len(selected) < count:
            start = generator.choice(sorted(remaining))
            width = 1
            if mode == "span":
                while width < 10 and generator.random() > .2:
                    width += 1
            for position in range(start, start + width):
                if position not in remaining or len(selected) >= count:
                    break
                selected.append(position)
                remaining.remove(position)
        for position in selected:
            labels[row, position] = inputs[row, position]
            draw = generator.random()
            if draw < .8:
                inputs[row, position] = tokenizer.mask_token_id
            elif draw < .9:
                inputs[row, position] = generator.randrange(len(tokenizer))
    if not (labels != -100).any():
        raise ValueError("MLM batch contains no maskable tokens")
    return {**batch, "input_ids": inputs.to(device), "labels": labels.to(device)}
