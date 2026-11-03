"""Dense retrieval or fixed-candidate reranking with explicit corpus scope."""
import argparse
import json
import time
import statistics
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from src.common import read_jsonl, write_json, sha256
from src.modeling import Encoder, pool
from src.metrics import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queries", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--model", default="jhu-clsp/mmBERT-base")
    p.add_argument("--revision")
    p.add_argument("--code-revision")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--chunk-length", type=int, help="Optional nonoverlapping chunk baseline, max score per document")
    p.add_argument("--scope", choices=["candidate_pool", "full_corpus"], required=True)
    p.add_argument("--candidates", help="Previous retrieval result JSON; required for reranking")
    p.add_argument("--rerank-k", type=int, default=50)
    p.add_argument("--split", choices=["dev", "test"], help="Required by the architecture selection protocol")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if Path(a.output).exists():
        raise ValueError("Use a new evaluation output path to preserve prior results")
    if not 10 <= a.rerank_k <= 100:
        raise ValueError("rerank-k must be between 10 and 100")
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    queries, documents = read_jsonl(a.queries), read_jsonl(a.corpus)
    ids = [d["docid"] for d in documents]
    id_set = set(ids)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate corpus document IDs")
    qrels = {q["query_id"]: {d["docid"]: 1 for d in q["positive_passages"]} for q in queries}
    if len(qrels) != len(queries):
        raise ValueError("Duplicate query IDs")
    if any(d not in id_set for relevant in qrels.values() for d in relevant):
        raise ValueError("A judged positive is missing from the corpus")
    if a.scope == "full_corpus" and "_pool" in Path(a.corpus).stem:
        raise ValueError("A candidate pool cannot be labeled full corpus")
    if a.checkpoint:
        cfg = json.loads((Path(a.checkpoint) / "config.json").read_text(encoding="utf-8"))
        model = Encoder(cfg)
        model.load_transfer(a.checkpoint, load_head=True)
        if cfg["task"] not in {"retrieval", "rerank"}:
            raise ValueError("Evaluation expects a retrieval or reranking checkpoint")
        model.to(device).eval()
        tokenize = model.tokenize
        embed = model.embed
        tokenizer = model.tokenizer
    else:
        cfg = {"task": "retrieval", "model": a.model, "revision": a.revision, "pooling": a.pooling}
        if a.trust_remote_code and (not a.revision or not a.code_revision):
            raise ValueError("Remote model code requires pinned --revision and --code-revision")
        tokenizer = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
        model = AutoModel.from_pretrained(a.model, revision=a.revision, code_revision=a.code_revision,
                                         trust_remote_code=a.trust_remote_code).to(device).eval()
        model.config.reference_compile = False
        def tokenize(texts, length):
            batch = tokenizer(texts, padding=True, truncation=True, max_length=length, return_tensors="pt")
            batch.pop("token_type_ids", None)
            return {k: v.to(device) for k, v in batch.items()}
        def embed(batch):
            output = model(**batch).last_hidden_state
            return F.normalize(output[:, 0], dim=-1) if a.pooling == "cls" else pool(output, batch["attention_mask"])
    if use_amp:
        torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    query_seconds = []
    candidate_recall = None
    runs = {}
    with torch.inference_mode(), torch.autocast(device_type=device, dtype=dtype, enabled=use_amp):
        if cfg["task"] == "rerank":
            if not a.candidates:
                raise ValueError("Reranking requires a fixed retrieval candidate file")
            candidate_result = json.loads(Path(a.candidates).read_text(encoding="utf-8"))
            if candidate_result["scope"] != a.scope:
                raise ValueError("Candidate scope differs from the reranking scope")
            if candidate_result["queries_sha256"] != sha256(a.queries) or candidate_result["corpus_sha256"] != sha256(a.corpus):
                raise ValueError("Reranking candidate file was generated on different queries or corpus")
            candidates = candidate_result["runs"]
            candidate_recall = evaluate(qrels, candidates, a.rerank_k)[0][f"recall@{a.rerank_k}"]
            lookup = {d["docid"]: d["text"] for d in documents}
            for q in queries:
                query_start = time.perf_counter()
                candidate_ids = candidates[q["query_id"]][:a.rerank_k]
                scores = []
                for docid in candidate_ids:
                    batch = model.tokenize([q["query"]], a.max_length, [lookup[docid]])
                    scores.append(float(model.score(batch)[0]))
                runs[q["query_id"]] = [candidate_ids[i] for i in sorted(range(len(scores)), key=lambda i: -scores[i])]
                query_seconds.append(time.perf_counter() - query_start)
        elif a.chunk_length:
            from src.interventions import special_template
            prefix, suffix = special_template(tokenizer)
            width = a.chunk_length - len(prefix) - len(suffix)
            if width < 1 or a.chunk_length > a.max_length:
                raise ValueError("Invalid chunk length")
            vectors = torch.cat([embed(tokenize([q["query"]], 128)).float() for q in queries])
            columns = []
            for document in documents:
                ids_raw = tokenizer.encode(document["text"], add_special_tokens=False)[:a.max_length-len(prefix)-len(suffix)]
                best = torch.full((len(queries),), -torch.inf, device=device)
                for offset in range(0, max(1, len(ids_raw)), width):
                    chunk = torch.tensor([prefix + ids_raw[offset:offset+width] + suffix], device=device)
                    encoded = embed({"input_ids": chunk, "attention_mask": torch.ones_like(chunk)}).float()
                    best = torch.maximum(best, (vectors @ encoded.T).squeeze(-1))
                columns.append(best.cpu())
            score_matrix = torch.stack(columns, dim=1)
            for i, q in enumerate(queries):
                ranking = torch.argsort(score_matrix[i], descending=True, stable=True)[:100].tolist()
                runs[q["query_id"]] = [ids[j] for j in ranking]
        else:
            embeddings = []
            for index in range(0, len(documents), a.batch_size):
                texts = [d["text"] for d in documents[index:index + a.batch_size]]
                embeddings.append(embed(tokenize(texts, a.max_length)).float().cpu())
                if index % 1000 == 0:
                    print(f"Encoded {index}/{len(documents)} documents", flush=True)
            matrix = torch.cat(embeddings)
            for q in queries:
                query_start = time.perf_counter()
                vector = embed(tokenize([q["query"]], 128)).float().cpu()
                scores = (vector @ matrix.T)[0]
                indices = torch.argsort(scores, descending=True, stable=True)[:100].tolist()
                runs[q["query_id"]] = [ids[i] for i in indices]
                query_seconds.append(time.perf_counter() - query_start)
    scores, per_query = evaluate(qrels, runs, 10)
    recall_k = a.rerank_k if cfg["task"] == "rerank" else 100
    recall, _ = evaluate(qrels, runs, recall_k)
    scores[f"recall@{recall_k}"] = recall[f"recall@{recall_k}"]
    if candidate_recall is not None:
        scores[f"candidate_recall@{a.rerank_k}"] = candidate_recall
    write_json(a.output, {"metrics": scores, "per_query": per_query, "runs": runs,
                          "scope": a.scope, "corpus_size": len(ids), "config": cfg,
                          "max_length": a.max_length, "pooling": a.pooling,
                          "checkpoint": a.checkpoint,
                          "split": a.split, "rerank_k": a.rerank_k,
                          "candidates_sha256": sha256(a.candidates) if a.candidates else None,
                          "architecture_sha256": sha256(Path(a.checkpoint) / "architecture.pt") if a.checkpoint and (Path(a.checkpoint) / "architecture.pt").exists() else None,
                          "adapter_sha256": sha256(Path(a.checkpoint) / "adapter/adapter_model.safetensors") if a.checkpoint else None,
                          "chunk_length": a.chunk_length,
                          "query_latency_median_seconds": statistics.median(query_seconds) if query_seconds else None,
                          "query_latency_p95_seconds": sorted(query_seconds)[min(len(query_seconds)-1, int(.95*len(query_seconds)))] if query_seconds else None,
                          "seconds": time.monotonic() - start,
                          "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30 if use_amp else 0,
                          "queries_sha256": sha256(a.queries), "corpus_sha256": sha256(a.corpus)})
    print(json.dumps(scores))


if __name__ == "__main__":
    main()
