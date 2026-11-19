import os
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMClient:
    """Local open-source chat LLM client (HuggingFace transformers).

    Supports:
      - chat completion via __call__(messages)
      - dense embedding via encode(texts)
      - per-item relevance probability via score_relevance(user_text, item_texts)
    """

    def __init__(
        self,
        model,
        api_key=None,
        base_url=None,
        temperature=0.0,
        max_tokens=2048,
        timeout=120,
        device=None,
        dtype=None,
        cache_dir=None,
        trust_remote_code=True,
        use_chat_template=True,
    ):
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_chat_template = use_chat_template
        self.timeout = timeout

        # ---- device / dtype auto-detection ----
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        if dtype is None:
            if str(device).startswith("cuda"):
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                dtype = torch.float32
        self.dtype = dtype

        # ---- load tokenizer + model ----
        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        _tf_ver = tuple(int(x) for x in transformers.__version__.split(".")[:2])
        _dtype_key = "dtype" if _tf_ver >= (4, 56) else "torch_dtype"

        load_kwargs = dict(
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            **{_dtype_key: dtype},
        )
        if device == "cuda":
            load_kwargs["device_map"] = "auto"
            self.model = AutoModelForCausalLM.from_pretrained(model, **load_kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model, **load_kwargs).to(device)

        self.model.eval()

    # ------------------------------------------------------------------
    # Chat completion (kept for backward compatibility)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def __call__(self, messages):
        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            encoded = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if hasattr(encoded, "input_ids"):
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                else:
                    attention_mask = torch.ones_like(input_ids)
            else:
                input_ids = encoded.to(self.device)
                attention_mask = torch.ones_like(input_ids)
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
            enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_ids = enc.input_ids
            attention_mask = enc.attention_mask

        do_sample = self.temperature is not None and self.temperature > 0.0
        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.05,
            no_repeat_ngram_size=4,
        )
        if do_sample:
            gen_kwargs["temperature"] = self.temperature

        output = self.model.generate(**gen_kwargs)
        new_tokens = output[0, input_ids.shape[1]:]
        content = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        if not content.strip():
            raise RuntimeError("The LLM returned an empty response.")
        return content.strip()

    # ------------------------------------------------------------------
    # Dense embedding (mean-pooled last hidden state)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def encode(self, texts, batch_size=8, max_length=512, pooling="mean"):
        """Encode texts into dense vectors using the LM's last hidden state.

        Args:
            texts: list[str]
            pooling: "mean" (masked mean) or "last" (last non-pad token)

        Returns:
            torch.Tensor of shape [len(texts), hidden_dim], dtype float32, on CPU.
        """
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)

            out = self.model(**enc, output_hidden_states=True)
            last_hidden = out.hidden_states[-1]                      # [B, T, D]
            mask = enc["attention_mask"].unsqueeze(-1).float()       # [B, T, 1]

            if pooling == "mean":
                pooled = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            elif pooling == "last":
                seq_lens = enc["attention_mask"].sum(1) - 1
                pooled = last_hidden[
                    torch.arange(last_hidden.size(0), device=last_hidden.device),
                    seq_lens,
                ]
            else:
                raise ValueError(f"Unknown pooling: {pooling}")

            all_embs.append(pooled.float().cpu())

        return torch.cat(all_embs, dim=0)

    # ------------------------------------------------------------------
    # Per-item relevance probability via Yes/No logits
    # ------------------------------------------------------------------
    def _yes_no_ids(self):
        """Resolve single-token ids for Yes / No across tokenizer variants."""
        yes_ids, no_ids = set(), set()
        for s in ["Yes", " Yes", "yes", " yes"]:
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            if len(ids) == 1:
                yes_ids.add(ids[0])
        for s in ["No", " No", "no", " no"]:
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            if len(ids) == 1:
                no_ids.add(ids[0])
        if not yes_ids or not no_ids:
            raise RuntimeError(
                f"Cannot resolve single-token Yes/No for tokenizer of {self.model_name}"
            )
        return sorted(yes_ids), sorted(no_ids)

    @torch.inference_mode()
    def score_relevance(self, user_text, item_texts, batch_size=4, max_length=1024):
        if not item_texts:
            return torch.zeros(0, dtype=torch.float32)

        yes_ids, no_ids = self._yes_no_ids()
        yes_t = torch.tensor(yes_ids, device=self.device)
        no_t = torch.tensor(no_ids, device=self.device)

        system = (
            "You are a recommendation system. Given a user's profile and watch history, "
            "decide whether a candidate movie is likely to be watched and enjoyed by the user. "
            "Answer with one word only: Yes or No."
        )

        probs = []
        for i in range(0, len(item_texts), batch_size):
            batch = item_texts[i : i + batch_size]
            # Build chat-formatted prompts so the model enters "assistant" mode
            prompts = []
            for item in batch:
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content":
                        f"{user_text}\n\nCandidate:\n{item}\n\nIs this candidate relevant to the user?"},
                ]
                prompts.append(
                    self.tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False
                    )
                )

            enc = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)

            try:
                out = self.model(**enc, logits_to_keep=1)
                last_logits = out.logits[:, -1, :]
            except TypeError:
                out = self.model(**enc)
                seq_lens = enc["attention_mask"].sum(1) - 1
                bsz = enc["input_ids"].size(0)
                last_logits = out.logits[
                    torch.arange(bsz, device=self.device), seq_lens
                ]

            yes_logit = last_logits.index_select(1, yes_t).max(dim=-1).values
            no_logit = last_logits.index_select(1, no_t).max(dim=-1).values
            pair = torch.stack([no_logit, yes_logit], dim=-1)
            p = torch.softmax(pair.float(), dim=-1)[:, 1]
            probs.append(p.cpu())

        return torch.cat(probs, dim=0)