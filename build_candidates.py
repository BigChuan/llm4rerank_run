#!/usr/bin/env python3
"""ML-1M flat records -> per-user samples with positive + negative candidates.

For each user:
  - history_items : up to `--history-max` most recent interactions (not in candidates)
  - candidate_items: `n_pos` most recent interactions (score preserved)
                   + `n_neg` random never-interacted movies (score = 0)
  - candidate order is shuffled so position doesn't leak the label

Output schema (matches Reranking.update_data):
  {
    "user_info":       {...},
    "history_items":   [{...}],
    "candidate_items": [{"item_id": ..., "score": ..., ...}],
    "meta":            {"user_id": ..., "n_pos": ..., "n_neg": ..., "n_total": ...}
  }
"""
import json
import random
import argparse
from collections import defaultdict
from pathlib import Path


def load(path):
    with open(path, encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="Flat per-rating JSON/JSONL produced by ml1m_to_json.py")
    ap.add_argument("--output", required=True)
    ap.add_argument("--history-max", type=int, default=20)
    ap.add_argument("--n-pos", type=int, default=5,
                    help="Positive candidates per user (most recent N interactions)")
    ap.add_argument("--n-neg", type=int, default=50,
                    help="Negative candidates per user (random unseen movies)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--user-fields", default="user_id,age,occupation")
    ap.add_argument("--item-fields", default="item_id,title,genre,year,score")
    args = ap.parse_args()

    user_fields = [x.strip() for x in args.user_fields.split(",") if x.strip()]
    item_fields = [x.strip() for x in args.item_fields.split(",") if x.strip()]
    # score must exist in item_fields; force-append if user forgot
    if "score" not in item_fields:
        item_fields.append("score")

    # ---------- load ----------
    records = load(args.input)
    print(f"[load] {len(records)} rating records")

    # ---------- build movie table ----------
    movies = {}
    for r in records:
        mid = r["item_id"]
        if mid not in movies:
            movies[mid] = {k: r.get(k) for k in ["item_id", "title", "genre", "year"]}
    all_movie_ids = sorted(movies.keys())
    print(f"[load] {len(all_movie_ids)} unique movies")

    # ---------- group by user ----------
    by_user = defaultdict(list)
    for r in records:
        by_user[r["user_id"]].append(r)
    print(f"[load] {len(by_user)} users")

    samples = []
    skipped = 0
    for uid, recs in by_user.items():
        recs.sort(key=lambda x: x.get("timestamp", 0))

        if len(recs) < 2:
            skipped += 1
            continue

        # history: everything except the last n_pos; truncated to history_max from the end
        hist_end = max(0, len(recs) - args.n_pos)
        hist_recs = recs[:hist_end]
        if args.history_max and len(hist_recs) > args.history_max:
            hist_recs = hist_recs[-args.history_max:]

        # positive candidates: last n_pos interactions
        pos_recs = recs[hist_end:]

        # negative candidates: random unseen movies
        watched = {r["item_id"] for r in recs}
        neg_pool = [m for m in all_movie_ids if m not in watched]
        user_rng = random.Random(args.seed + uid)
        n_neg = min(args.n_neg, len(neg_pool))
        neg_ids = user_rng.sample(neg_pool, n_neg) if n_neg > 0 else []

        # build candidate list
        candidates = []
        for r in pos_recs:
            cand = {k: r.get(k) for k in item_fields}
            if cand.get("score") is None:
                cand["score"] = 0.0
            candidates.append(cand)
        for mid in neg_ids:
            cand = {k: movies[mid].get(k) for k in item_fields}
            cand["score"] = 0.0         # negative label
            candidates.append(cand)

        # shuffle so position doesn't leak the label
        user_rng.shuffle(candidates)

        sample = {
            "user_info": {k: recs[0].get(k) for k in user_fields},
            "history_items": [
                {k: r.get(k) for k in item_fields} for r in hist_recs
            ],
            "candidate_items": candidates,
            "meta": {
                "user_id": uid,
                "n_history": len(hist_recs),
                "n_pos": len(pos_recs),
                "n_neg": n_neg,
                "n_total": len(candidates),
            },
        }
        samples.append(sample)

    # ---------- save ----------
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    # ---------- stats ----------
    n_total = len(samples)
    avg_pool = sum(s["meta"]["n_total"] for s in samples) / max(1, n_total)
    avg_pos = sum(s["meta"]["n_pos"] for s in samples) / max(1, n_total)
    avg_neg = sum(s["meta"]["n_neg"] for s in samples) / max(1, n_total)
    avg_hist = sum(s["meta"]["n_history"] for s in samples) / max(1, n_total)
    print(f"\n[save] {n_total} samples -> {out.resolve()}")
    print(f"[stat] skipped (users with <2 ratings): {skipped}")
    print(f"[stat] avg pool size  : {avg_pool:.1f}")
    print(f"[stat] avg n_pos      : {avg_pos:.1f}")
    print(f"[stat] avg n_neg      : {avg_neg:.1f}")
    print(f"[stat] avg n_history  : {avg_hist:.1f}")

    # sanity preview
    s0 = samples[0]
    print(f"\n[preview] user {s0['meta']['user_id']}:")
    print(f"  history   : {len(s0['history_items'])} items")
    print(f"  candidates: {len(s0['candidate_items'])} items "
          f"(pos={s0['meta']['n_pos']}, neg={s0['meta']['n_neg']})")
    print(f"  first 2 candidates:")
    for c in s0["candidate_items"][:2]:
        print(f"    {c}")


if __name__ == "__main__":
    main()