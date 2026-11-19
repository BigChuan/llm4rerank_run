
"""LLM encoding + MLP relevance reranking."""

import torch

from .llm_encoder import LLMEncoder
from .mlp_reranker import MLPReranker


class Reranking:

    def __init__(
        self,
        dataset_name,
        llm,
        user_fea=None,
        his_item_fea=None,
        item_fea=None,
        top_k=10,
        history_max=20,
        preselect_ratio=3.0,
        mlp_hidden=128,
        mlp_layers=2,
        mlp_lr=1e-3,
        lambda_div=0.0,
        diversity_temperature=0.2,
        device=None,
        **legacy_kwargs,
    ):
        self.dataset_name = dataset_name.lower()
        self.llm = llm

        self.top_k = top_k
        self.history_max = history_max
        self.preselect_ratio = preselect_ratio

        self.mlp_hidden = mlp_hidden
        self.mlp_layers = mlp_layers
        self.mlp_lr = mlp_lr
        self.lambda_div = lambda_div
        self.diversity_temperature = diversity_temperature

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.encoder = LLMEncoder(
            llm_client=llm,
            user_features=user_fea or [],
            history_features=his_item_fea or [],
            item_features=item_fea or [],
            dataset_name=dataset_name,
            history_max=history_max,
        )

        self.mlp = None
        self._mlp_trained = False

        self.data = None
        self.candidate_id = []

    @staticmethod
    def _id_key(v):
        return str(v).strip()

    def update_data(self, data):
        required = {
            "user_info",
            "history_items",
            "candidate_items",
        }

        missing = required - set(data)

        if missing:
            raise ValueError(
                f"Input data missing keys: {sorted(missing)}"
            )

        candidates = data["candidate_items"]
        history = data["history_items"]

        if hasattr(candidates, "to_dict"):
            candidates = candidates.to_dict("records")

        if hasattr(history, "to_dict"):
            history = history.to_dict("records")

        if not candidates:
            raise ValueError("candidate_items is empty.")

        if not all("item_id" in x for x in candidates):
            raise ValueError(
                "Every candidate must contain item_id."
            )

        ids = [x["item_id"] for x in candidates]

        if len({str(x) for x in ids}) != len(ids):
            raise ValueError(
                "candidate_items contains duplicate item_id values."
            )

        self.data = {
            "user_info": data["user_info"],
            "history_items": history,
            "candidate_items": candidates,
        }

        self.candidate_id = ids

    def _encode_sample(self):
        user_emb, item_embs, relevance = self.encoder.encode_sample(
            self.data["user_info"],
            self.data["history_items"],
            self.data["candidate_items"],
        )

        self.user_emb = user_emb
        self.item_embs = item_embs
        self.relevance = relevance

        if self.mlp is None:
            input_dim = item_embs.size(1) + 1

            self.mlp = MLPReranker(
                input_dim=input_dim,
                top_k=self.top_k,
                hidden_dim=self.mlp_hidden,
                num_layers=self.mlp_layers,
                lr=self.mlp_lr,
                lambda_div=self.lambda_div,
                temperature=self.diversity_temperature,
                device=self.device,
            )

    def _preselect(self):
        n = len(self.candidate_id)

        top_n = min(
            n,
            max(
                int(self.top_k * self.preselect_ratio),
                self.top_k,
            ),
        )

        return LLMEncoder.preselect(
            self.relevance, top_n
        )

    def _build_mlp_input(self, idx):
        embeddings = self.item_embs[idx]
        relevance = self.relevance[idx].unsqueeze(-1)

        features = torch.cat(
            [embeddings, relevance], dim=-1
        )

        return features, embeddings

    def _labels_for(self, idx):
        labels = []

        for i in idx:
            value = self.data["candidate_items"][i].get(
                "score", 0.0
            )

            try:
                value = float(value)
            except (TypeError, ValueError):
                value = 0.0

            labels.append(
                1.0 if value >= 4.0 else 0.0
            )

        return torch.tensor(
            labels,
            dtype=torch.float32,
            device=self.device,
        )

    def fit(self, train_data, epochs=5, log_every=1):
        n = len(train_data)

        if n == 0:
            raise ValueError("Training data is empty.")

        for epoch in range(epochs):
            total_loss = 0.0
            total_acc = 0.0
            total_div = 0.0
            successful = 0

            for i, sample in enumerate(train_data):
                try:
                    self.update_data(sample)
                    self._encode_sample()

                    idx = self._preselect()

                    features, embeddings = (
                        self._build_mlp_input(idx)
                    )

                    labels = self._labels_for(idx)

                    loss, acc, div = self.mlp.train_step(
                        features.to(self.device),
                        labels,
                        embeddings.to(self.device),
                    )

                    total_loss += loss
                    total_acc += acc
                    total_div += div
                    successful += 1

                except Exception as exc:
                    print(
                        f"[fit] sample {i} skipped: {exc}",
                        flush=True,
                    )

            if successful == 0:
                raise RuntimeError(
                    "No training samples were processed successfully."
                )

            if (epoch + 1) % log_every == 0:
                print(
                    f"[fit] epoch {epoch+1}/{epochs} "
                    f"loss={total_loss/successful:.4f} "
                    f"acc={total_acc/successful:.4f} "
                    f"div={total_div/successful:.4f} "
                    f"ok={successful}/{n}",
                    flush=True,
                )

        self._mlp_trained = True

    @torch.no_grad()
    def rerank(self, data):
        self.update_data(data)
        self._encode_sample()

        idx = self._preselect()

        features, _ = self._build_mlp_input(idx)
        features = features.to(self.device)

        if self._mlp_trained:
            logits = self.mlp.score(features).cpu()

            mlp_prob = torch.sigmoid(logits)
            llm_prob = self.relevance[idx].float().cpu()

            final_scores = 0.8 * mlp_prob + 0.2 * llm_prob
        else:
            final_scores = (
                self.relevance[idx].float().cpu()
            )

        order = final_scores.argsort(
            descending=True
        ).tolist()

        preselected_ids = [
            self.candidate_id[idx[i]]
            for i in order
        ]

        final_ids = preselected_ids[:self.top_k]

        full_scores = [
            float("-inf")
        ] * len(self.candidate_id)

        for rank, i in enumerate(order):
            full_scores[idx[i]] = float(
                final_scores[i].item()
            )

        history = [
            ["llm_preselect", "mlp_rank"],
            [
                [self.candidate_id[i] for i in idx],
                preselected_ids,
            ],
        ]

        return final_ids, history, full_scores
