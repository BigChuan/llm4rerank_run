cd ~/llm4rerank_runner
cat > diag_fusion.py <<'EOF'
import torch
from llm4rerank import Reranking
from llm4rerank.llm_client import LLMClient
from eval_rerank import ndcg_at_k, load_json

data = load_json('data/ml-1m-pool-test.json')
llm = LLMClient(model='./models/llama-3.1-8b-instruct', temperature=0.0, max_tokens=2048)

r = Reranking(
    dataset_name='ml-1m', llm=llm,
    user_fea=['user_id','age','occupation'],
    his_item_fea=['item_id','title','genre','year'],
    item_fea=['item_id','title','genre','year','score'],
    top_k=10, history_max=20, preselect_ratio=3.0,
    mlp_hidden=128, mlp_layers=2, mlp_lr=1e-3,
    lambda_div=0.0, diversity_temperature=0.2,
)
r.update_data(data[0]); r._encode_sample()
r.mlp.model.load_state_dict(torch.load('checkpoints/mlp_m1_nodiv.pt', map_location=r.device))
r.mlp.model.eval()

def ndcg_of(order_scores_fn, sample):
    r.update_data(sample); r._encode_sample()
    idx = r._preselect()
    feats, _ = r._build_mlp_input(idx)
    feats = feats.to(r.device)
    with torch.no_grad():
        logits = r.mlp.score(feats).cpu()
    mlp_prob = torch.sigmoid(logits)
    llm_prob = r.relevance[idx].float().cpu()
    final = order_scores_fn(mlp_prob, llm_prob)
    order = final.argsort(descending=True).tolist()
    ranking = [r.candidate_id[idx[i]] for i in order][:10]
    id2score = {c['item_id']: float(c.get('score', 0)) for c in sample['candidate_items']}
    return ndcg_at_k(ranking, id2score, 10)

fns = {
    '纯 LLM (0/1)':      lambda m, l: l,
    '纯 MLP (1/0)':      lambda m, l: m,
    '0.5/0.5 (当前)':    lambda m, l: 0.5*m + 0.5*l,
    '0.8/0.2':           lambda m, l: 0.8*m + 0.2*l,
    '0.9/0.1':           lambda m, l: 0.9*m + 0.1*l,
    'logit 融合':         lambda m, l: torch.logit(m.clamp(1e-6,1-1e-6)) + l,
}
for name, fn in fns.items():
    vals = [ndcg_of(fn, s) for s in data]
    print(f"{name:20s} ndcg@10 = {sum(vals)/len(vals):.4f}")
EOF
python diag_fusion.py