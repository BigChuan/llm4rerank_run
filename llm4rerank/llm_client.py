import os
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMClient:
    """Local open-source chat LLM client (HuggingFace transformers).

    Downloads the model from the HF Hub on first use (cached afterwards).
    Maintains the same interface as the OpenAI version:
        client = LLMClient(model="Qwen/llama-3.1-8b-instruct")
        text = client(messages)   # messages = [{"role": ..., "content": ...}, ...]
    """

    def __init__(
        self,
        model,
        api_key=None,            # kept for signature compatibility; unused
        base_url=None,           # kept for signature compatibility; unused
        temperature=0.0,
        max_tokens=2048,
        timeout=120,             # unused locally, kept for compatibility
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
                # bf16 if supported, else fp16
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                dtype = torch.float32
        self.dtype = dtype

        # ---- load tokenizer + model (downloads on first run) ----
        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
        # Ensure pad token exists (needed for batched generate / some models)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # transformers >= 4.56 renamed `torch_dtype` -> `dtype`
        _tf_ver = tuple(int(x) for x in transformers.__version__.split(".")[:2])
        _dtype_key = "dtype" if _tf_ver >= (4, 56) else "torch_dtype"

        load_kwargs = dict(
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            **{_dtype_key: dtype},
        )
        # device_map="auto" only makes sense for plain "cuda" (multi-GPU / offload);
        # explicit "cuda:1" etc. goes through manual .to() to stay on that card.
        if device == "cuda":
            load_kwargs["device_map"] = "auto"
            self.model = AutoModelForCausalLM.from_pretrained(model, **load_kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model, **load_kwargs).to(device)

        self.model.eval()

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def __call__(self, messages):
        """messages: list[dict] with 'role' and 'content' (OpenAI chat format)."""
        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            encoded = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            # New transformers returns BatchEncoding; old returns a bare Tensor.
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
            # Fallback: naive concatenation (only for models without a chat template)
            prompt = "\n".join(
                f"{m['role']}: {m['content']}" for m in messages
            ) + "\nassistant:"
            enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_ids = enc.input_ids
            attention_mask = enc.attention_mask

        # temperature == 0  -> greedy
        do_sample = self.temperature is not None and self.temperature > 0.0
        # gen_kwargs = dict(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     max_new_tokens=self.max_tokens,
        #     do_sample=do_sample,
        #     pad_token_id=self.tokenizer.pad_token_id,
        #     eos_token_id=self.tokenizer.eos_token_id,
        # )
        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.05,          # ← 新增：抑制复读
            no_repeat_ngram_size=4,           # ← 新增：禁止 4-gram 重复
        )
        if do_sample:
            gen_kwargs["temperature"] = self.temperature

        output = self.model.generate(**gen_kwargs)

        # Slice off the prompt tokens (only return the newly generated part)
        new_tokens = output[0, input_ids.shape[1]:]
        content = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        if not content.strip():
            raise RuntimeError("The LLM returned an empty response.")
        return content.strip()