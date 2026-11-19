
#!/usr/bin/env python3
"""Train the MLP reranker on labeled samples and save a checkpoint.

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
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            data = [json.loads(line) for line in f if line.strip()]

    return data


def compute_ndcg_at_k(ranks, k=10):
    """Compute mean NDCG@K using raw item scores as graded relevance."""
    import math

    def dcg(rels):
        return sum(
            (2 ** rel - 1) / math.log2(i + 2)
            for i, rel in enumerate(rels)
        )

    scores = []

    for sample, ranking in ranks:
        candidates = sample["candidate_items"]

        id2score = {
            item["item_id"]: float(item.get("score", 0))
            for item in candidates
        }

        rels = [
            id2score.get(item_id, 0.0)
            for item_id in ranking[:k]
        ]

        ideal = sorted(id2score.values(), reverse=True)[:k]

        ideal_dcg = dcg(ideal)
        scores.append(
            dcg(rels) / ideal_dcg if ideal_dcg > 0 else 0.0
        )

    return sum(scores) / len(scores) if scores else 0.0


def main():
    ap = argparse.ArgumentParser(
        description="Train an MLP-based LLM reranker."
    )

    # Data and model
    ap.add_argument("--train-input", required=True)
    ap.add_argument(
        "--val-input",
        default=None,
        help="Optional validation set."
    )
    ap.add_argument(
        "--output-ckpt",
        required=True,
        help="Path to save the trained MLP state_dict."
    )
    ap.add_argument("--dataset", default="ml-1m")
    ap.add_argument(
        "--model",
        default="./models/llama-3.1-8b-instruct"
    )

    # Features
    ap.add_argument(
        "--user-features",
        default="user_id,age,occupation"
    )
    ap.add_argument(
        "--history-features",
        default="item_id,title,genre,year"
    )
    ap.add_argument(
        "--item-features",
        default="item_id,title,genre,year,score"
    )

    # Reranking configuration
    ap.add_argument("--history-max", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--preselect-ratio", type=float, default=3.0)

    # MLP hyperparameters
    ap.add_argument("--mlp-hidden", type=int, default=128)
    ap.add_argument("--mlp-layers", type=int, default=2)
    ap.add_argument("--mlp-lr", type=float, default=1e-3)

    # Optional differentiable diversity regularization
    ap.add_argument("--lambda-div", type=float, default=0.0)
    ap.add_argument(
        "--diversity-temperature",
        type=float,
        default=0.2
    )

    # Training
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--log-every", type=int, default=1)

    args = ap.parse_args()

    if args.epochs <= 0:
        ap.error("--epochs must be positive")
    if args.top_k <= 0:
        ap.error("--top-k must be positive")
    if args.mlp_hidden <= 0 or args.mlp_layers <= 0:
        ap.error("--mlp-hidden and --mlp-layers must be positive")
    if args.mlp_lr <= 0:
        ap.error("--mlp-lr must be positive")
    if args.diversity_temperature <= 0:
        ap.error("--diversity-temperature must be positive")

    # ---------- data ----------
    train = load_json(args.train_input)

    if isinstance(train, dict):
        train = [train]

    if not train:
        raise ValueError("Training dataset is empty.")

    print(f"[data] train samples = {len(train)}", flush=True)

    val = None
    if args.val_input:
        val = load_json(args.val_input)

        if isinstance(val, dict):
            val = [val]

        print(f"[data] val samples   = {len(val)}", flush=True)

    # ---------- LLM + MLP reranker ----------
    llm = LLMClient(
        model=args.model,
        temperature=0.0,
        max_tokens=2048,
    )

    reranker = Reranking(
        dataset_name=args.dataset,
        llm=llm,
        user_fea=parse_features(args.user_features),
        his_item_fea=parse_features(args.history_features),
        item_fea=parse_features(args.item_features),
        top_k=args.top_k,
        history_max=args.history_max,
        preselect_ratio=args.preselect_ratio,
        mlp_hidden=args.mlp_hidden,
        mlp_layers=args.mlp_layers,
        mlp_lr=args.mlp_lr,
        lambda_div=args.lambda_div,
        diversity_temperature=args.diversity_temperature,
    )

    # ---------- train ----------
    print(
        f"[train] architecture=MLP "
        f"epochs={args.epochs} "
        f"hidden={args.mlp_hidden} "
        f"layers={args.mlp_layers} "
        f"lr={args.mlp_lr} "
        f"lambda_div={args.lambda_div} "
        f"top_k={args.top_k}",
        flush=True,
    )

    reranker.fit(
        train,
        epochs=args.epochs,
        log_every=args.log_every,
    )

    # ---------- save checkpoint ----------
    ckpt = Path(args.output_ckpt)
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        reranker.mlp.model.state_dict(),
        ckpt,
    )

    print(f"[save] MLP checkpoint -> {ckpt.resolve()}", flush=True)

    # ---------- optional validation ----------
    if val is not None:
        print(
            f"[val] evaluating on {len(val)} samples",
            flush=True,
        )

        reranker._mlp_trained = True
        reranker.mlp.model.eval()

        ranks = []

        for i, sample in enumerate(val):
            try:
                ranking, _, _ = reranker.rerank(sample)
                ranks.append((sample, ranking))
            except Exception as e:
                print(
                    f"[val] sample {i} failed: {e}",
                    flush=True,
                )

        ndcg = compute_ndcg_at_k(ranks, k=args.top_k)

        print(
            f"[val] NDCG@{args.top_k} = {ndcg:.4f} "
            f"({len(ranks)}/{len(val)} samples)",
            flush=True,
        )


if __name__ == "__main__":
    main()
