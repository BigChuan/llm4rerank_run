"""GNN scorer on the preselected item subgraph.

Node features: [LLM embedding (D) || LLM relevance prob (1)]
Loss: BCE(score, label) + lambda * diversity_regularizer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp_model import MLPModel


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, edge_index, edge_weight):
        N = x.size(0)
        row, col = edge_index[0], edge_index[1]
        deg = torch.zeros(N, device=x.device).scatter_add_(0, row, edge_weight)
        dinv = deg.clamp(min=1e-6).pow(-0.5)
        nw = edge_weight * dinv[row] * dinv[col]
        agg = torch.zeros_like(x)
        agg.index_add_(0, row, x[col] * nw.unsqueeze(-1))
        return self.W(agg)


class ItemGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([GCNLayer(hidden_dim, hidden_dim)
                                     for _ in range(num_layers)])
        self.scorer = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight):
        x = self.in_proj(x)
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.scorer(x).squeeze(-1)


class GraphBuilder:
    @staticmethod
    def knn_graph(embs, k=6):
        N = embs.size(0)
        if N <= 1:
            loops = torch.arange(N, device=embs.device)
            return torch.stack([loops, loops], 0), torch.ones(N, device=embs.device)
        e = F.normalize(embs.float(), dim=-1)
        sim = e @ e.T
        sim.fill_diagonal_(-float("inf"))
        kk = min(k, N - 1)
        _, knn = sim.topk(kk, dim=-1)
        src = torch.arange(N, device=embs.device).unsqueeze(1).expand(-1, kk).reshape(-1)
        dst = knn.reshape(-1)
        w = sim[src, dst]
        ei = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], 0)
        ew = torch.cat([w, w])
        loops = torch.arange(N, device=embs.device)
        ei = torch.cat([ei, torch.stack([loops, loops])], 1)
        ew = torch.cat([ew, torch.ones(N, device=embs.device)])
        return ei, ew


class GNNReranker:
    def __init__(self, node_dim, top_k=10, hidden_dim=128, num_layers=2,
                 lr=1e-3, lambda_div=0.3, device=None, arch="mlp", **kw):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if arch == "mlp":
            self.model = MLPModel(node_dim, hidden_dim, num_layers,
                                  dropout=kw.get("dropout", 0.1),
                                  use_residual=kw.get("use_residual", True)).to(self.device)
        else:
            self.model = ItemGNN(node_dim, hidden_dim, num_layers).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr,
                                     weight_decay=kw.get("weight_decay", 1e-4))
        self.top_k = top_k
        self.lambda_div = lambda_div
        self.arch = arch
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.top_k = top_k
        self.lambda_div = lambda_div

    def _diversity_reg(self, scores, embs):
        k = min(self.top_k, scores.numel())
        if k <= 1:
            return torch.zeros((), device=scores.device)
        topk_idx = scores.topk(k).indices
        v = F.normalize(embs[topk_idx].float(), dim=-1)
        sim = v @ v.T
        mask = ~torch.eye(k, dtype=torch.bool, device=sim.device)
        return sim[mask].mean()

    def train_step(self, node_x, edge_index, edge_weight, labels, embs):
        self.model.train()
        self.opt.zero_grad()
        scores = self.model(node_x, edge_index, edge_weight)
        acc_loss = F.binary_cross_entropy_with_logits(scores, labels)
        div_loss = self._diversity_reg(scores, embs)
        loss = acc_loss + self.lambda_div * div_loss
        loss.backward()
        self.opt.step()
        return loss.item(), acc_loss.item(), div_loss.item()

    @torch.no_grad()
    def score(self, node_x, edge_index, edge_weight):
        self.model.eval()
        return self.model(node_x, edge_index, edge_weight)