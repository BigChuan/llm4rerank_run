import argparse
import json
import os
import traceback
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(
        description="Run the WWW'25 LLM4Rerank reference algorithm."
    )
    parser.add_argument("--input", required=True, help="Input JSON/JSONL file.")
    parser.add_argument("--output", default="rerank_result.json")
    parser.add_argument("--dataset", default="ml-1m",
                        help="Dataset name, e.g. ml-1m or kuairand.")
    parser.add_argument("--model", default=os.getenv("LOCAL_MODEL", "./models/llama-3.1-8b-instruct"),
                        help="Local model path or HF repo id.")
    parser.add_argument("--api-key", default=None,
                        help="Optional; unused for local models.")
    parser.add_argument("--base-url", default=None,
                        help="Optional; unused for local models.")
    parser.add_argument("--focus", default="Overall Performance")
    parser.add_argument("--nodes", default="diversity,accuracy,fairness,stop")
    parser.add_argument("--user-features", default="")
    parser.add_argument("--history-features", default="")
    parser.add_argument("--item-features", default="item_id,score")
    parser.add_argument("--history-max", type=int, default=5)
    parser.add_argument("--max-count", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    data = load_json(args.input)

    llm = LLMClient(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    nodes = parse_features(args.nodes)
    reranker = Reranking(
        dataset_name=args.dataset,
        llm=llm,
        nodes=nodes,
        user_fea=parse_features(args.user_features),
        his_item_fea=parse_features(args.history_features),
        item_fea=parse_features(args.item_features),
        top_k=args.top_k,
        focus=args.focus,
        history_max=args.history_max,
        max_count=args.max_count,
        max_retries=args.max_retries,
    )

    def get_user_id(sample):
        ui = sample.get("user_info") or {}
        return ui.get("user_id", sample.get("user_id"))


    def rerank_one(sample):
        uid = get_user_id(sample)
        candidates = get_candidate_items(sample)
        try:
            ranking, history, score = reranker.rerank(sample)
            return {
                "user_id": uid,
                "candidate_item_ids": [x["item_id"] for x in candidates],
                "final_ranking": ranking,
                "operation_history": history[0],
                "ranking_history": history[1],
                "score": score,
                "top_k": ranking[:args.top_k],
                "fallback": False,
            }
        except Exception as e:
            return {
                "user_id": uid,
                "candidate_item_ids": [x["item_id"] for x in candidates],
                "final_ranking": [x["item_id"] for x in candidates],   # 原始顺序兜底
                "error": f"{type(e).__name__}: {e}",
                "fallback": True,
            }

    if isinstance(data, list):
        results = []
        total = len(data)
        for i, sample in enumerate(data):
            try:
                results.append(rerank_one(sample))
            except Exception as e:
                print(f"[{i}] user={sample.get('user_id')} failed: {e}", flush=True)
                traceback.print_exc()
                results.append({
                    "user_id": sample.get("user_id"),
                    "error": f"{type(e).__name__}: {e}",
                })
            if (i + 1) % 10 == 0 or (i + 1) == total:
                print(f"[{i+1}/{total}] done", flush=True)

        result = {
            "dataset": args.dataset,
            "model": args.model,
            "num_samples": total,
            "results": results,
        }
    else:
        result = {
            "dataset": args.dataset,
            "model": args.model,
            **rerank_one(data),
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2)[:2000], "...")
    print("\nSaved result to:", output_path.resolve())


if __name__ == "__main__":
    main()