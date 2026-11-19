import json, torch
from llm4rerank import Reranking
from llm4rerank.llm_client import LLMClient
from eval_rerank import ndcg_at_k, load_json

data = load_json('data/ml-1m-pool-test.json')
llm = LLMClient(model='./models/llama-3.1-8b-instruct', temperature=0.0, max_tokens=2048)

def make_reranker():
    return Reranking(
        dataset_name='ml-1m', llm=llm,
        user_fea=['user_id','age','occupation'],
        his_item_fea=['item_id','title','genre','year'],
        item_fea=['item_id','title','genre','year','score'],
        top_k=10, history_max=20, preselect_ratio=3.0,
        gnn_hidden=128, gnn_layers=2, gnn_lr=1e-3,
        lambda_div=0.0, k_neighbors=0,
        mlp_hidden=128, mlp_layers=2, mlp_lr=1e-3,   # ← 若 __init__ 不接受会报错，删掉这行
    )

def eval_ckpt(ckpt_path, arch):
    r = make_reranker()
    first = data[0]
    r.update_data(first)
    r._encode_sample()
    state = torch.load(ckpt_path, map_location=r.device)
    if arch == 'mlp':
        r.mlp.model.load_state_dict(state)
        r.mlp.model.eval()
        r._mlp_trained = True
    else:
        r.gnn.model.load_state_dict(state)
        r.gnn.model.eval()
        r._gnn_trained = True
    scores = []
    for s in data:
        ranking, _, _ = r.rerank(s)
        id2score = {c['item_id']: float(c.get('score', 0)) for c in s['candidate_items']}
        scores.append(ndcg_at_k(ranking, id2score, 10))
    return sum(scores) / len(scores)

print(f"GNN k=0 (旧 0.4991): {eval_ckpt('checkpoints/gnn_ml1m_div0.3_k0.pt', 'gnn'):.4f}")
print(f"MLP nodiv (旧 0.4513): {eval_ckpt('checkpoints/mlp_m1_nodiv.pt', 'mlp'):.4f}")
