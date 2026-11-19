"""LLM-side encoder: user/item embeddings + per-item relevance probability."""
import torch
import torch.nn.functional as F


class LLMEncoder:
    """Produces (user_emb, item_embs, relevance_probs) for one sample."""

    def __init__(self, llm_client, user_features, history_features, item_features,
                 dataset_name="ml-1m", history_max=10):
        self.llm = llm_client
        self.user_features = list(user_features or [])
        self.history_features = list(history_features or [])
        self.item_features = list(item_features or [])
        self.dataset_name = dataset_name.lower()
        self.history_max = history_max

    # ---------------- formatting ----------------
    def _format_item(self, item):
        parts = []
        for f in self.item_features:
            if f == "score":           # drop ground-truth label from the prompt
                continue
            v = item.get(f)
            if v is None:
                continue
            if isinstance(v, list):
                v = "|".join(str(x) for x in v)
            parts.append(f"{f}={v}")
        return "Item " + "; ".join(parts) if parts else "Item"

    def _format_history(self, history_items):
        lines = ["- " + self._format_item(h) for h in history_items[: self.history_max]]
        return "\n".join(lines) if lines else "(empty)"

    def _format_user(self, user_info, history_items):
        u = []
        for f in self.user_features:
            v = user_info.get(f)
            if v is not None:
                u.append(f"{f}={v}")
        return (
            "User profile: " + "; ".join(u)
            + "\nWatched history:\n" + self._format_history(history_items)
        )

    # ---------------- encoding ----------------
    def encode_items(self, items):
        return self.llm.encode([self._format_item(it) for it in items])   # [N, D]

    def encode_user(self, user_info, history_items):
        text = self._format_user(user_info, history_items)
        return self.llm.encode([text])[0]                                  # [D]

    def score_relevance(self, user_info, history_items, items):
        user_text = self._format_user(user_info, history_items)
        item_texts = [self._format_item(it) for it in items]
        return self.llm.score_relevance(user_text, item_texts)            # [N]

    # ---------------- unified call ----------------
    def encode_sample(self, user_info, history_items, items):
        """Return user_emb, item_embs, relevance_probs."""
        user_emb = self.encode_user(user_info, history_items)
        item_embs = self.encode_items(items)
        rel = self.score_relevance(user_info, history_items, items)
        return user_emb, item_embs, rel

    # ---------------- preselect ----------------
    @staticmethod
    @torch.no_grad()
    def preselect(relevance_probs, top_n):
        """Keep the top-N items by LLM relevance probability."""
        n = min(top_n, relevance_probs.numel())
        idx = relevance_probs.topk(n).indices.tolist()
        return idx