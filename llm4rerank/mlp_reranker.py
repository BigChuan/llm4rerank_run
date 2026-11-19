
import torch
import torch.nn.functional as F

from .mlp_model import MLPModel


class MLPReranker:

    def __init__(
        self,
        input_dim,
        top_k=10,
        hidden_dim=128,
        num_layers=2,
        lr=1e-3,
        lambda_div=0.0,
        temperature=0.2,
        dropout=0.1,
        weight_decay=1e-4,
        device=None,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.top_k = top_k
        self.lambda_div = lambda_div
        self.temperature = temperature

        self.model = MLPModel(
            in_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    def diversity_loss(self, logits, embeddings):
        n = logits.numel()
        k = min(self.top_k, n)

        if k <= 1:
            return logits.new_zeros(())

        # Similarity is computed from fixed LLM item embeddings.
        with torch.no_grad():
            emb = F.normalize(
                embeddings.detach().float(), dim=-1
            )
            sim = emb @ emb.T

            mask = ~torch.eye(
                n, dtype=torch.bool, device=logits.device
            )

        # Detached kth-largest threshold.
        threshold = logits.detach().topk(k).values[-1]

        # Differentiable soft selection.
        weights = torch.sigmoid(
            (logits - threshold) /
            max(self.temperature, 1e-6)
        )

        pair_weights = weights[:, None] * weights[None, :]

        numerator = (
            pair_weights * sim
        ).masked_select(mask).sum()

        denominator = pair_weights.masked_select(mask).sum()

        return numerator / denominator.clamp(min=1e-8)

    def train_step(self, features, labels, embeddings):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        logits = self.model(features)

        relevance_loss = F.binary_cross_entropy_with_logits(
            logits, labels
        )

        div_loss = self.diversity_loss(
            logits, embeddings
        )

        loss = (
            relevance_loss
            + self.lambda_div * div_loss
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 5.0
        )

        self.optimizer.step()

        with torch.no_grad():
            acc = (
                (logits >= 0).float() == labels
            ).float().mean()

        return (
            loss.item(),
            acc.item(),
            div_loss.item(),
        )

    @torch.no_grad()
    def score(self, features):
        self.model.eval()
        return self.model(features)
