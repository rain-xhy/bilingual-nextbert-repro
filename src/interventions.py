"""Token-exact controlled interventions. Context is evidence-bearing, not a minimal rationale."""
import hashlib
import unicodedata


VIEWS = ("front", "middle", "back", "negative", "deleted", "sham")


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def context_id(text):
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def context_split(text):
    # All questions about the same source context belong to the same split.
    return "dev" if int(context_id(text)[:8], 16) % 10 == 0 else "train"


def make_views(evidence, filler, replacement, length):
    if not evidence or len(evidence) * 3 > length or len(filler) < length or len(replacement) < len(evidence):
        raise ValueError("Need nonempty evidence <= one third of the budget and sufficient non-evidence tokens")
    n = len(evidence)
    background = list(filler[:length - n])
    replacement = list(replacement[:n])
    starts = {"front": 0, "middle": len(background) // 2, "back": len(background)}
    views = {name: background[:cut] + list(evidence) + background[cut:] for name, cut in starts.items()}
    middle = starts["middle"]
    views["negative"] = list(filler[:length])
    views["deleted"] = background[:middle] + replacement + background[middle:]
    # Replace an equal-size irrelevant span, keeping the evidence and length unchanged.
    sham_start = 0 if middle >= n else middle + n
    views["sham"] = list(views["middle"])
    views["sham"][sham_start:sham_start+n] = replacement
    if any(len(v) != length for v in views.values()):
        raise AssertionError("Interventions must have identical token lengths")
    return views, {name: [start, start + n] for name, start in starts.items()}


def special_template(tokenizer):
    count_special = tokenizer.num_special_tokens_to_add(pair=False)
    # Fast tokenizers may implement special-token insertion in the backend postprocessor,
    # while build_inputs_with_special_tokens() remains the no-op base implementation.
    template = tokenizer("a", add_special_tokens=True, return_special_tokens_mask=True)
    content = [i for i, mask in enumerate(template["special_tokens_mask"]) if not mask]
    if not content:
        raise ValueError("Cannot infer the single-sequence special-token template")
    prefix = template["input_ids"][:content[0]]
    suffix = template["input_ids"][content[-1]+1:]
    if len(prefix) + len(suffix) != count_special:
        raise ValueError("Unsupported interleaved special-token template")
    return prefix, suffix


def materialize(row, tokenizer, device):
    import torch
    prefix, suffix = special_template(tokenizer)
    views, _ = make_views(row["evidence_ids"], row["filler_ids"], row["replacement_ids"],
                         row["total_length"] - len(prefix) - len(suffix))
    sequences = [prefix + row["query_ids"] + suffix]
    sequences += [prefix + views[key] + suffix for key in VIEWS]
    if any(len(ids) != row["total_length"] for ids in sequences[1:]):
        raise ValueError("Document materialization failed the exact-length invariant")
    batches = []
    for ids in sequences:
        if len(ids) > row["total_length"]:
            raise ValueError("Query or document exceeds the declared token length")
        tensor = torch.tensor([ids], device=device)
        batches.append({"input_ids": tensor, "attention_mask": torch.ones_like(tensor)})
    return batches
