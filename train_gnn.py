#!/usr/bin/env python3
"""Train the GNN reranker on labeled samples and save a checkpoint.

Relevance label: score >= 4 -> positive (1.0), else negative (0.0).
"""
import argparse
import json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-input", required=True)
    ap.add_argument("--val-input", default=None,
                    help="Optional validation set; skipped if omitted.")
    ap.add_argument("--output-ckpt", required=True,
                    help="Where to save the trained GNN state_dict.")
    ap.add_argument("--dataset", default="ml-1m")
    ap.add_argument("--model", default="./models/llama-3.1-8b-instruct")

    ap.add_argument("--user-features", default="user_id,age,occupation")
    ap.add_argument("--history-features", default="item_id,title,genre,year")
    ap.add_argument("--item-features", default="item_id,title,genre,year,score")
    ap.add_argument("--history-max", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=10)

    # GNN hyperparams
    ap.add_argument("--preselect-ratio", type=float, default=3.0)
    ap.add_argument("--gnn-hidden", type=int, default=128)
    ap.add_argument("--gnn-layers", type=int, default=2)
    ap.add_argument("--gnn-lr", type=float, default=1e-3)
    ap.add_argument("--lambda-div", type=float, default=0.3)
    ap.add_argument("--k-neighbors", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--log-every", type=int, default=1)
    args = ap.parse_args()

    # ---------- data ----------
    train = load_json(args.train_input)
    if not isinstance(train, list):
        train = [train]
    print(f"[data] train samples = {len(train)}", flush=True)
    if args.val_input:
        val = load_json(args.val_input)
        if not isinstance(val, list):
            val = [val]
        print(f"[data] val samples   = {len(val)}", flush=True)

    # ---------- LLM + reranker ----------
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

    # ---------- train ----------
    print(f"[train] epochs={args.epochs}  lr={args.gnn_lr}  "
          f"lambda_div={args.lambda_div}  k={args.k_neighbors}", flush=True)
    reranker.fit(train, epochs=args.epochs, log_every=args.log_every)

    # ---------- save ----------
    ckpt = Path(args.output_ckpt)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(reranker.gnn.model.state_dict(), ckpt)
    print(f"[save] -> {ckpt.resolve()}", flush=True)

    # ---------- optional val ----------
    if args.val_input:
        print(f"[val] evaluating on {len(val)} samples", flush=True)
        reranker._gnn_trained = True
        ranks = []
        for i, sample in enumerate(val):
            try:
                ranking, _, _ = reranker.rerank(sample)
                ranks.append((sample, ranking))
            except Exception as e:
                print(f"[val] sample {i} failed: {e}", flush=True)
        ndcg = compute_ndcg_at_k(ranks, k=args.top_k)
        print(f"[val] NDCG@{args.top_k} = {ndcg:.4f}", flush=True)


# ---------- inline NDCG helper (train-time sanity check only) ----------
def compute_ndcg_at_k(ranks, k=10):
    """relevance = raw score (0-5), normalized by ideal DCG."""
    import math
    def dcg(rels):
        return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))
    scores = []
    for sample, ranking in ranks:
        id2score = {x["item_id"]: float(x.get("score", 0)) for x in sample["candidate_items"]}
        rels = [id2score.get(x, 0.0) for x in ranking[:k]]
        ideal = sorted(id2score.values(), reverse=True)[:k]
        scores.append(dcg(rels) / dcg(ideal) if dcg(ideal) > 0 else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


if __name__ == "__main__":
    main()