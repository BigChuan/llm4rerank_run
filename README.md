# LLM4Rerank runnable wrapper

Runnable wrapper around the supplied WWW'25 LLM4Rerank reference snippet.
It adds an OpenAI-compatible LLM client, CLI, JSON input/output, response
validation, bounded retries, and operation/ranking history.

## Install

```bash
conda create -n llm4rerank python=3.10 -y
conda activate llm4rerank
pip install -r requirements.txt
```

Or install the requirements into an existing Python 3.8+ environment.

## Configure the API

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
export OPENAI_MODEL="gpt-4o-mini"
```

For an OpenAI-compatible endpoint, also set:

```bash
export OPENAI_BASE_URL="https://YOUR_ENDPOINT/v1"
```

The original `Request_LLM()` is explicitly a privacy placeholder; replace
the client if your provider uses a different API protocol. Never commit API keys.

## Input schema

See `example_input.json`. Required top-level keys:

```json
{
  "user_info": {},
  "history_items": [],
  "candidate_items": [
    {"item_id": 1, "score": 0.9}
  ]
}
```

Each candidate must have a unique `item_id`. Add item attributes required by
the objectives, such as `genre`, `year`, `upload_type`, or `video_duration`.

## Run

```bash
python run.py \
  --input example_input.json \
  --output outputs/result.json \
  --dataset ml-1m \
  --model gpt-4o-mini \
  --nodes diversity,accuracy,fairness,stop \
  --user-features user_id,age,occupation \
  --history-features item_id,title,genre,year \
  --item-features item_id,title,genre,year,score \
  --focus "Overall Performance" \
  --history-max 5 \
  --max-count 3 \
  --top-k 10
```

Output JSON contains `final_ranking`, `operation_history`, `ranking_history`,
`score` (zero-based final position for each candidate in original candidate
order), and `top_k`.

## Objectives and scope

Supported nodes: `accuracy`, `diversity`, `fairness`, `stop`; the controller
may also request `backward` to undo the latest accepted operation.

The supplied reference gives dataset-specific diversity/fairness wording for
`ml-1m` and `kuairand`. For other datasets this wrapper uses generic attribute-
based instructions; those generic instructions are wrapper defaults, not claims
about the paper's exact experimental protocol.

This is an execution wrapper, not the complete official repository or a full
paper reproduction. The supplied snippet does not include dataset preprocessing,
candidate generation, base recommender scoring, or evaluation scripts. You must
provide candidate items and their features in the JSON schema above.

The reference's `top_k` argument is unused in its ranking logic. Here it only
controls the saved `top_k` output; the LLM reranks the full candidate list.
Start with a small candidate list to check prompt size and API cost.
