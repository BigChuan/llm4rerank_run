
import torch
import torch.nn as nn
import torch.nn.functional as F


# class ResBlock(nn.Module):
#     def __init__(self, d, p=0.1):
#         super().__init__()
#         self.norm = nn.LayerNorm(d)
#         self.net = nn.Sequential(
#             nn.Linear(d, d * 2),
#             nn.GELU(),
#             nn.Dropout(p),
#             nn.Linear(d * 2, d),
#             nn.Dropout(p),
#         )

#     def forward(self, x):
#         return x + self.net(self.norm(x))


# class MLPModel(nn.Module):
#     """Pure pointwise MLP scorer."""

#     def __init__(
#         self,
#         in_dim,
#         hidden_dim=128,
#         num_layers=2,
#         dropout=0.1,
#         use_residual=True,
#     ):
#         super().__init__()

#         self.in_proj = nn.Linear(in_dim, hidden_dim)
#         self.norm = nn.LayerNorm(hidden_dim)
#         self.drop = nn.Dropout(dropout)

#         self.residual = use_residual and num_layers > 1

#         if self.residual:
#             self.blocks = nn.ModuleList([
#                 ResBlock(hidden_dim, dropout)
#                 for _ in range(num_layers - 1)
#             ])
#         else:
#             self.blocks = nn.ModuleList([
#                 nn.Sequential(
#                     nn.Linear(hidden_dim, hidden_dim),
#                     nn.GELU(),
#                     nn.Dropout(dropout),
#                 )
#                 for _ in range(num_layers - 1)
#             ])

#         self.scorer = nn.Linear(hidden_dim, 1)

#     def forward(self, x):
#         h = self.drop(
#             self.norm(F.gelu(self.in_proj(x)))
#         )

#         for block in self.blocks:
#             h = block(h)

#         return self.scorer(h).squeeze(-1)

class MLPModel(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_layers=2, dropout=0.1, **kw):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scorer = nn.Linear(hidden_dim, 1)
        self.drop = nn.Dropout(dropout)
    def forward(self, x, edge_index=None, edge_weight=None):
        h = self.in_proj(x)
        h = self.hidden(h)
        h = F.relu(h)
        h = self.drop(h)
        return self.scorer(h).squeeze(-1)   # ← scorer 前无激活，关键