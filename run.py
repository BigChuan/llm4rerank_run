import argparse
import json
import os
import traceback
from pathlib import Path

import torch

from llm4rerank import Reranking
from llm4rerank.llm_client import LLMClient


def parse_features(value):
    if value is None or value.strip() == "":
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)                       # array / single object
        except json.JSONDecodeError:
            f.seek(0)
            return [json.loads(l) for l in f if l.strip()]   # jsonl


def get_candidate_items(sample):
    """兼容 candidate_items / candidates 两种字段名。"""
    if "candidate_items" in sample:
        return sample["candidate_items"]
    if "candidates" in sample:
        return sample["candidates"]
    raise KeyError(
        f"sample has no candidates field; got keys: {sorted(sample.keys())}"
    )


def get_user_id(sample):
    ui = sample.get("user_info") or {}
    return ui.get("user_id", sample.get("user_id"))


def _ensure_gnn(reranker, sample):
    """Instantiate reranker.gnn by running one forward through the encoder.

    Reranking only constructs the GNN lazily inside _encode_sample(),
    so we need one sample to know the embedding dimension before load_state_dict.
    """
    if reranker.gnn is None:
        reranker.update_data(sample)
        reranker._encode_sample()
    return reranker.gnn


def load_gnn_checkpoint(reranker, ckpt_path, sample):
    _ensure_gnn(reranker, sample)
    state = torch.load(ckpt_path, map_location=reranker.device)
    reranker.gnn.model.load_state_dict(state)
    reranker.gnn.model.eval()
    reranker._gnn_trained = True         # ← 新增
    print(f"[init] loaded GNN weights from {ckpt_path}", flush=True)


def save_gnn_checkpoint(reranker, ckpt_path):
    if reranker.gnn is None:
        raise RuntimeError("Cannot save GNN checkpoint: gnn is not initialized.")
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(reranker.gnn.model.state_dict(), ckpt_path)
    print(f"[init] saved GNN weights to {ckpt_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="LLM-encode + GNN-rerank pipeline (accuracy-focused)."
    )
    # ---------- core IO ----------
    parser.add_argument("--input", required=True, help="Test/input JSON/JSONL file.")
    parser.add_argument("--output", default="rerank_result.json")
    parser.add_argument("--dataset", default="ml-1m",
                        help="Dataset name, e.g. ml-1m or kuairand.")
    parser.add_argument("--model", default=os.getenv("LOCAL_MODEL", "./models/llama-3.1-8b-instruct"),
                        help="Local model path or HF repo id.")

    # ---------- legacy (kept for compatibility, now ignored by Reranking) ----------
    parser.add_argument("--api-key", default=None, help="Unused for local models.")
    parser.add_argument("--base-url", default=None, help="Unused for local models.")
    parser.add_argument("--nodes", default="diversity,accuracy,fairness,stop",
                        help="(deprecated) old multi-node list; ignored in new pipeline.")
    parser.add_argument("--focus", default="Overall Performance",
                        help="(deprecated) old goal string; ignored.")
    parser.add_argument("--max-count", type=int, default=3,
                        help="(deprecated) old iteration cap; ignored.")
    parser.add_argument("--max-retries", type=int, default=5,
                        help="(deprecated) old LLM retry count; ignored.")

    # ---------- features ----------
    parser.add_argument("--user-features", default="")
    parser.add_argument("--history-features", default="")
    parser.add_argument("--item-features", default="item_id,score")
    parser.add_argument("--history-max", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)

    # ---------- LLM decoding (encode / score_relevance) ----------
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)

    # ---------- NEW: GNN pipeline ----------
    parser.add_argument("--preselect-ratio", type=float, default=3.0,
                        help="How many candidates to keep for GNN: top_k * ratio.")
    parser.add_argument("--gnn-hidden", type=int, default=128)
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--gnn-lr", type=float, default=1e-3)
    parser.add_argument("--lambda-div", type=float, default=0.3,
                        help="Weight of diversity regularizer vs accuracy BCE.")
    parser.add_argument("--k-neighbors", type=int, default=6,
                        help="k for the kNN item graph.")
    parser.add_argument("--gnn-ckpt", default=None,
                        help="Path to GNN checkpoint (load if exists, save after training).")
    parser.add_argument("--train-input", default=None,
                        help="Optional training JSON/JSONL; runs GNN.fit() before inference.")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs when --train-input is given.")

    args = parser.parse_args()

    # ---------- data ----------
    data = load_json(args.input)
    samples = data if isinstance(data, list) else [data]
    first_sample = samples[0]

    # ---------- LLM ----------
    llm = LLMClient(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # ---------- Reranking ----------
    # New API: nodes/focus/max_count/max_retries accepted but ignored.
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
        # legacy kwargs (ignored by new Reranking, but accepted for signature compatibility)
        nodes=parse_features(args.nodes),
        focus=args.focus,
        max_count=args.max_count,
        max_retries=args.max_retries,
    )

    # ---------- GNN: load or train ----------
    if args.gnn_ckpt and os.path.exists(args.gnn_ckpt):
        load_gnn_checkpoint(reranker, args.gnn_ckpt, first_sample)
    elif args.train_input:
        train = load_json(args.train_input)
        if not isinstance(train, list):
            train = [train]
        print(f"[init] training GNN on {len(train)} samples for {args.epochs} epochs", flush=True)
        reranker.fit(train, epochs=args.epochs)
        if args.gnn_ckpt:
            save_gnn_checkpoint(reranker, args.gnn_ckpt)
    else:
        print("[init] no GNN checkpoint and no --train-input; "
              "falling back to LLM relevance probability as score", flush=True)

    # ---------- rerank one sample ----------
    def rerank_one(sample):
        uid = get_user_id(sample)
        candidates = get_candidate_items(sample)
        try:
            ranking, history, score = reranker.rerank(sample)
            # history[0]: ["llm_preselect", "gnn_rank"]
            # history[1]: [pre-selected ids (len ≈ top_k * ratio), final ranked ids]
            return {
                "user_id": uid,
                "candidate_item_ids": [x["item_id"] for x in candidates],
                "final_ranking": ranking,
                "operation_history": history[0],       # now fixed labels
                "ranking_history": history[1],          # [preselect, final]
                "score": score,                         # len == len(candidates)
                "top_k": ranking[:args.top_k],
                "fallback": False,
            }
        except Exception as e:
            traceback.print_exc()
            return {
                "user_id": uid,
                "candidate_item_ids": [x["item_id"] for x in candidates],
                "final_ranking": [x["item_id"] for x in candidates],   # original order fallback
                "error": f"{type(e).__name__}: {e}",
                "fallback": True,
            }

    # ---------- loop ----------
    if isinstance(data, list):
        results = []
        total = len(data)
        n_fallback = 0
        for i, sample in enumerate(data):
            r = rerank_one(sample)
            if r.get("fallback"):
                n_fallback += 1
            results.append(r)
            if (i + 1) % 10 == 0 or (i + 1) == total:
                print(f"[{i+1}/{total}] done  fallback={n_fallback}", flush=True)

        result = {
            "dataset": args.dataset,
            "model": args.model,
            "num_samples": total,
            "num_fallback": n_fallback,
            "results": results,
        }
    else:
        result = {
            "dataset": args.dataset,
            "model": args.model,
            **rerank_one(data),
        }

    # ---------- save ----------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    preview = json.dumps(result, ensure_ascii=False, indent=2)[:2000]
    print(preview, "...")
    print("\nSaved result to:", output_path.resolve())


if __name__ == "__main__":
    main()