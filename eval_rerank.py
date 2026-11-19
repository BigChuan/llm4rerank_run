#!/usr/bin/env python3
"""Evaluate a reranker on a held-out set with NDCG@K, MRR, and Diversity@K."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch

from llm4rerank import Reranking
from llm4rerank.llm_client import LLMClient


def parse_features(v):
    return [x.strip() for x in v.split(",") if x.strip()] if v else []


def load_json(path):
    with open(path, encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            return [json.loads(l) for l in f if l.strip()]


# ---------- metrics ----------
def dcg(rels):
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranking, id2score, k):
    rels = [id2score.get(x, 0.0) for x in ranking[:k]]
    ideal = sorted(id2score.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(rels) / idcg if idcg > 0 else 0.0


def mrr_at_k(ranking, id2score, k, pos_threshold=4):
    for i, x in enumerate(ranking[:k]):
        if id2score.get(x, 0.0) >= pos_threshold:
            return 1.0 / (i + 1)
    return 0.0


def hit_at_k(ranking, id2score, k, pos_threshold=4):
    return 1.0 if any(id2score.get(x, 0) >= pos_threshold for x in ranking[:k]) else 0.0


def diversity_at_k(ranking, id2genres, k):
    """Fraction of distinct genres / total genre labels in top-k (higher = more diverse)."""
    genres = []
    for x in ranking[:k]:
        genres.extend(id2genres.get(x, []))
    if not genres:
        return 0.0
    return len(set(genres)) / len(genres)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dataset", default="ml-1m")
    ap.add_argument("--model", default="./models/llama-3.1-8b-instruct")
    ap.add_argument("--user-features", default="user_id,age,occupation")
    ap.add_argument("--history-features", default="item_id,title,genre,year")
    ap.add_argument("--item-features", default="item_id,title,genre,year,score")
    ap.add_argument("--history-max", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--preselect-ratio", type=float, default=3.0)
    ap.add_argument("--gnn-hidden", type=int, default=128)
    ap.add_argument("--gnn-layers", type=int, default=2)
    ap.add_argument("--gnn-lr", type=float, default=1e-3)
    ap.add_argument("--lambda-div", type=float, default=0.3)
    ap.add_argument("--k-neighbors", type=int, default=6)
    ap.add_argument("--gnn-ckpt", default=None,
                    help="If given, load the checkpoint. Otherwise use LLM-only.")
    args = ap.parse_args()

    data = load_json(args.input)
    if not isinstance(data, list):
        data = [data]
    print(f"[data] samples = {len(data)}", flush=True)

    llm = LLMClient(model=args.model, temperature=0.0, max_tokens=2048)
    reranker = Reranking(
        dataset_name=args.dataset,
        llm=llm,
        user_fea=parse_features(args.user_features),
        his_item_fea=parse_features(args.history_features),
        item_fea=parse_features(args.item_features),
        top_k=args.top_k,
        history_max=args.history_max,
        preselect_ratio=args.preselect_ratio,
        gnn_hidden=args.gnn_hidden,
        gnn_layers=args.gnn_layers,
        gnn_lr=args.gnn_lr,
        lambda_div=args.lambda_div,
        k_neighbors=args.k_neighbors,
    )

    # ---------- load GNN if provided ----------
    if args.gnn_ckpt:
        first = data[0]
        reranker.update_data(first)
        reranker._encode_sample()          # builds gnn with correct in_dim
        state = torch.load(args.gnn_ckpt, map_location=reranker.device)
        reranker.gnn.model.load_state_dict(state)
        reranker.gnn.model.eval()
        reranker._gnn_trained = True
        print(f"[init] loaded GNN from {args.gnn_ckpt}", flush=True)
    else:
        print("[init] no --gnn-ckpt; using LLM relevance only", flush=True)

    # ---------- evaluate ----------
    results = []
    n_fail = 0
    for i, sample in enumerate(data):
        uid = (sample.get("user_info") or {}).get("user_id", sample.get("user_id"))
        cands = sample.get("candidate_items", [])
        try:
            ranking, history, scores = reranker.rerank(sample)
        except Exception as e:
            n_fail += 1
            results.append({"user_id": uid, "error": f"{type(e).__name__}: {e}"})
            continue

        id2score = {c["item_id"]: float(c.get("score", 0)) for c in cands}
        id2genres = {c["item_id"]: c.get("genre", []) for c in cands}
        # normalize genre into list
        for k, v in id2genres.items():
            if isinstance(v, str):
                id2genres[k] = [g for g in v.split("|") if g]

        # candidate order baseline
        cand_order = [c["item_id"] for c in cands]

        results.append({
            "user_id": uid,
            "metrics": {
                "ndcg@10":   ndcg_at_k(ranking, id2score, 10),
                "ndcg@5":    ndcg_at_k(ranking, id2score, 5),
                "mrr@10":    mrr_at_k(ranking, id2score, 10),
                "hit@10":    hit_at_k(ranking, id2score, 10),
                "div@10":    diversity_at_k(ranking, id2genres, 10),
            },
            "baseline": {
                "ndcg@10":   ndcg_at_k(cand_order, id2score, 10),
                "ndcg@5":    ndcg_at_k(cand_order, id2score, 5),
                "mrr@10":    mrr_at_k(cand_order, id2score, 10),
                "hit@10":    hit_at_k(cand_order, id2score, 10),
                "div@10":    diversity_at_k(cand_order, id2genres, 10),
            },
            "ranking": ranking,
        })
        if (i + 1) % 50 == 0 or (i + 1) == len(data):
            print(f"[{i+1}/{len(data)}] fail={n_fail}", flush=True)

    # ---------- aggregate ----------
    valid = [r for r in results if "metrics" in r]
    def mean(key, section="metrics"):
        vals = [r[section][key] for r in valid]
        return sum(vals) / len(vals) if vals else 0.0

    summary = {
        "num_samples": len(data),
        "num_valid": len(valid),
        "num_failed": n_fail,
        "model": "GNN" if args.gnn_ckpt else "LLM-only",
        "rerank":   {k: round(mean(k), 4) for k in ["ndcg@10", "ndcg@5", "mrr@10", "hit@10", "div@10"]},
        "baseline": {k: round(mean(k, "baseline"), 4) for k in ["ndcg@10", "ndcg@5", "mrr@10", "hit@10", "div@10"]},
        "delta": {
            k: round(mean(k) - mean(k, "baseline"), 4)
            for k in ["ndcg@10", "ndcg@5", "mrr@10", "hit@10", "div@10"]
        },
    }

    print("\n===== Summary =====")
    for section in ["rerank", "baseline", "delta"]:
        print(f"[{section}]")
        for k, v in summary[section].items():
            print(f"  {k:10s}: {v:+.4f}" if section == "delta" else f"  {k:10s}: {v:.4f}")

    # ---------- save ----------
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n[save] -> {out.resolve()}")


if __name__ == "__main__":
    main()