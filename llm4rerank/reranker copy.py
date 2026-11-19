import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_NODES = ["diversity", "accuracy", "stop"]


class Reranking:
    """
    Iterative LLM reranker adapted from the supplied WWW'25 reference code.

    Input data format:
      {
        "user_info": {...},
        "history_items": [{...}, ...],
        "candidate_items": [{ "item_id": ..., "score": ..., ... }, ...]
      }

    The LLM must return:
      [item_id_1, item_id_2, ...]; next_operation

    where next_operation is one of the configured nodes or "backward".
    """

    def __init__(
        self,
        dataset_name: str,
        llm,
        nodes: Optional[Sequence[str]] = None,
        user_fea: Optional[Sequence[str]] = None,
        his_item_fea: Optional[Sequence[str]] = None,
        item_fea: Optional[Sequence[str]] = None,
        top_k: int = 10,
        focus: str = "Overall Performance",
        history_max: int = 5,
        max_count: int = 3,
        max_retries: int = 5,
    ):
        self.dataset_name = dataset_name.lower()
        self.llm = llm
        self.nodes = list(nodes or DEFAULT_NODES)
        self.user_fea = list(user_fea or [])
        self.his_item_fea = list(his_item_fea or [])
        self.item_fea = list(item_fea or [])
        self.top_k = top_k
        self.focus = focus
        self.history_max = history_max
        self.max_count = max_count
        self.max_retries = max_retries
        self.re_history = [[], []]
        self.data = None
        self.candidate_id = []

        if "stop" not in self.nodes:
            self.nodes.append("stop")
        if len(set(self.nodes)) != len(self.nodes):
            raise ValueError("The nodes list contains duplicates.")
        allowed = {"accuracy", "diversity", "fairness", "stop"}
        unknown = set(self.nodes) - allowed
        if unknown:
            raise ValueError("Unsupported reranking nodes: {}".format(sorted(unknown)))

    def update_data(self, data: Dict[str, Any]):
        required = {"user_info", "history_items", "candidate_items"}
        missing = required - set(data)
        if missing:
            raise ValueError("Input data missing keys: {}".format(sorted(missing)))

        candidates = data["candidate_items"]
        if hasattr(candidates, "to_dict"):
            candidates = candidates.to_dict("records")
        history = data["history_items"]
        if hasattr(history, "to_dict"):
            history = history.to_dict("records")

        if not candidates:
            raise ValueError("candidate_items is empty.")
        if not all("item_id" in row for row in candidates):
            raise ValueError("Every candidate must contain item_id.")

        ids = [row["item_id"] for row in candidates]
        if len({str(x) for x in ids}) != len(ids):
            raise ValueError("candidate_items contains duplicate item_id values.")

        self.data = {
            "user_info": data["user_info"],
            "history_items": history,
            "candidate_items": candidates,
        }
        self.candidate_id = ids

    @staticmethod
    def _id_key(value):
        return str(value).strip()

    @staticmethod
    def _display_id(value):
        # Keep numeric item IDs numeric in the prompt/output.
        s = str(value).strip()
        try:
            return int(s)
        except (TypeError, ValueError):
            return s

    def _call_llm(self, prompt, previous=None, correction=None):
        messages = [{"role": "user", "content": prompt}]
        if previous is not None:
            messages.append({"role": "assistant", "content": previous})
        if correction:
            messages.append({"role": "user", "content": correction})
        return self.llm(messages)

    def _parse_response(self, raw: str, require_ranking: bool):
        text = raw.strip()
        if not require_ranking:
            # The backward/initial decision is a single operation word.
            match = re.search(r"\b(accuracy|diversity|fairness|stop)\b", text.lower())
            if not match:
                raise ValueError("Could not parse the requested next operation.")
            operation = match.group(1)
            return None, operation

        list_match = re.search(r"\[([^\]]*)\]", text)
        if not list_match:
            raise ValueError("Response does not contain a bracketed item ID list.")
        # raw_ids = [x.strip().strip("'\"") for x in list_match.group(1).split(",") if x.strip()]
        raw_ids = [x.strip().strip("'\"") for x in list_match.group(1).split(",") if x.strip()]
        # 清理末尾的 "; operation" 污染
        raw_ids = [re.sub(r"[;\s].*$", "", x) for x in raw_ids]
        raw_ids = [x for x in raw_ids if x]
        if not raw_ids:
            raise ValueError("The reranking list is empty.")

        # Parse operation after the closing bracket; tolerate punctuation/formatting.
        suffix = text[list_match.end():].lower()
        op_match = re.search(r"\b(accuracy|diversity|fairness|stop|backward)\b", suffix)
        if not op_match:
            raise ValueError("Response does not contain a valid next operation.")
        operation = op_match.group(1)

        candidate_keys = [self._id_key(x) for x in self.candidate_id]
        response_keys = [self._id_key(x) for x in raw_ids]
        lookup = {self._id_key(x): x for x in self.candidate_id}

        # 1) 只保留候选集内的合法 ID
        in_candidates = [k for k in response_keys if k in lookup]

        # 2) 合法 ID 太少 -> 判定为幻觉，直接失败重试
        if len(in_candidates) < max(1, int(0.6 * len(candidate_keys))):
            raise ValueError(
                "Too few valid candidate IDs ({} / {}); likely hallucinated.".format(
                    len(in_candidates), len(candidate_keys)
                )
            )

        # 3) 去重，保留首次出现顺序
        seen, deduped = set(), []
        for k in in_candidates:
            if k not in seen:
                seen.add(k)
                deduped.append(k)

        # 4) 补齐缺失的候选 ID（按候选原始顺序）
        for k in candidate_keys:
            if k not in seen:
                seen.add(k)
                deduped.append(k)

        # 5) 截断到候选数量
        response_keys = deduped[:len(candidate_keys)]

        # Restore the original Python type for each ID.
        lookup = {self._id_key(x): x for x in self.candidate_id}
        ranking = [lookup[k] for k in response_keys]
        if operation not in self.nodes and operation != "backward":
            raise ValueError("Operation '{}' is not in configured nodes {}.".format(
                operation, self.nodes
            ))
        return ranking, operation

    def request(self, prompt: str, current_name: str):
        is_backward_decision = current_name == "backward"
        base_temperature = getattr(self.llm, "temperature", None)
        last_raw = None
        last_exc = None
        current_prompt = prompt
    
        try:
            for attempt in range(self.max_retries):
                # On retries, raise temperature so greedy decoding doesn't repeat the same mistake.
                if attempt > 0 and base_temperature is not None:
                    self.llm.temperature = max(base_temperature, 0.7)

                raw = self._call_llm(current_prompt)
                last_raw = raw

                try:
                    ranking, operation = self._parse_response(
                        raw, require_ranking=not is_backward_decision
                    )
                except Exception as exc:
                    last_exc = exc
                    if attempt + 1 >= self.max_retries:
                        raise RuntimeError(
                            "LLM response validation failed after {} attempts. "
                            "Last response: {!r}. Last error: {}".format(
                                self.max_retries, last_raw, last_exc
                            )
                        ) from last_exc

                    # Feed the error back into the prompt and actually ask again.
                    correction = (
                        "Your previous response was invalid: {}. Please answer again. "
                        "For reranking, return exactly a permutation of all candidate item_id values "
                        "inside square brackets, followed by '; ' and one operation from: {}. "
                        "Do not include any other text.".format(
                            exc, ", ".join(self.nodes + ["backward"])
                        )
                    )
                    current_prompt = prompt + "\n\n" + correction
                    continue

                # Parse succeeded -> restore temperature before walking to the next node.
                if base_temperature is not None:
                    self.llm.temperature = base_temperature

                if is_backward_decision:
                    if operation == "backward":
                        operation = self.nodes[0] if self.nodes[0] != "stop" else "accuracy"
                    return getattr(self, operation + "_rerank")()

                self.re_history[0].append(current_name)
                self.re_history[1].append(ranking)

                if len(self.re_history[0]) >= self.max_count:
                    operation = "stop"
                if operation == "backward":
                    return self.backward_rerank()
                return getattr(self, operation + "_rerank")()

            # Should be unreachable: loop either returns or raises.
            raise RuntimeError(
                "LLM response validation failed after {} attempts. Last response: {!r}. "
                "Last error: {}".format(self.max_retries, last_raw, last_exc)
            )
        finally:
            if base_temperature is not None:
                self.llm.temperature = base_temperature

    def _build_context(self):
        if self.data is None:
            raise RuntimeError("Call update_data(data) before reranking.")

        user_info = self.data["user_info"]
        history = self.data["history_items"][:self.history_max]
        candidates = self.data["candidate_items"]

        def select_features(row, features):
            if not features:
                return row
            return {k: row.get(k) for k in features}

        context = {
            "user_info": select_features(user_info, self.user_fea),
            "history_items": [select_features(x, self.his_item_fea) for x in history],
            "candidate_items": [
                select_features(x, self.item_fea) for x in candidates
            ],
            "note": "'score' is generated by an existing recommendation model for reference.",
            "previous_operations": [
                {
                    "focus": focus,
                    "ranking": ranking,
                }
                for focus, ranking in zip(self.re_history[0], self.re_history[1])
            ],
        }
        return json.dumps(context, ensure_ascii=False, indent=2, default=str)

    def _history_instruction(self):
        if not self.re_history[0]:
            return "No historical reranking operations have been performed yet."
        return (
            "Use the previous operations and rankings in the supplied context. "
            "Do not discard them unless you explicitly choose backward."
        )

    def _make_ranking_prompt(self, focus_instruction: str):
        return (
            "You are an expert recommendation reranker.\n"
            "User, browsing history, candidate items, and previous reranking operations "
            "are provided in the JSON context below.\n\n"
            + self._build_context()
            + "\n\n"
            + self._history_instruction()
            + "\n\n"
            + focus_instruction
            + "\n\nThe output order must represent recommendation priority (most likely "
              "to be watched first). Return every candidate item_id exactly once, with no "
              "new IDs and no omissions. Then choose the next operation from: "
            + ", ".join(self.nodes)
            + ", backward. 'stop' ends reranking and returns the current ranking. "
              "'backward' deletes the most recent reranking operation and its result.\n"
            "Strict output format:\n[item_id_1, item_id_2, ...]; next_operation\n"
            "Do not include explanations or any other text."
        )

    def backward_rerank(self):
        # If invoked after an LLM-selected backward operation, undo the latest
        # accepted reranking step before asking the controller what to do next.
        if self.re_history[0]:
            self.re_history[0].pop()
            self.re_history[1].pop()

        # The reference method asks the LLM to choose the first/next focus.
        prompt = (
            "You are controlling an iterative recommendation reranking process.\n"
            + self._build_context()
            + "\nChoose the next reranking focus from: "
            + ", ".join([n for n in self.nodes if n != "stop"])
            + (", stop" if self.re_history[0] else "")
            + ". 'backward' means undo the most recent operation; it is unavailable "
              "when there is no previous operation.\n"
            "Final goal: " + self.focus + "\n"
            "Reply with only one operation name."
        )
        return self.request(prompt, "backward")

    def accuracy_rerank(self):
        instruction = (
            "Focus on ACCURACY: estimate the match between the user and each candidate "
            "using the user profile, browsing history, item features, and recommendation score. "
            "Rank items by how likely the user is to watch them. Choose the next focus based "
            "on the overall goal: " + self.focus + "."
        )
        return self.request(self._make_ranking_prompt(instruction), "accuracy")

    def diversity_rerank(self):
        if self.dataset_name == "ml-1m":
            detail = "Promote a variety of movie genres near the top of the list."
        elif self.dataset_name == "kuairand":
            detail = "Promote a variety of upload_type values near the top of the list."
        else:
            detail = "Promote meaningful variety across the available item attributes."
        instruction = (
            "Focus on DIVERSITY: " + detail
            + " Maintain reasonable relevance to the user. Choose the next focus based "
              "on the overall goal: " + self.focus + "."
        )
        return self.request(self._make_ranking_prompt(instruction), "diversity")

    def fairness_rerank(self):
        if self.dataset_name == "ml-1m":
            detail = (
                "Balance the average ranking positions of items released after 1996 "
                "and items released before 1996, using the year feature."
            )
        elif self.dataset_name == "kuairand":
            detail = (
                "Balance average ranking positions across groups defined by "
                "video_duration."
            )
        else:
            detail = (
                "Avoid systematically disadvantaging identifiable item groups. "
                "Use available item attributes to reason about group exposure."
            )
        instruction = (
            "Focus on FAIRNESS: " + detail
            + " Preserve recommendation relevance where possible. Choose the next focus "
              "based on the overall goal: " + self.focus + "."
        )
        return self.request(self._make_ranking_prompt(instruction), "fairness")

    def stop_rerank(self):
        if not self.re_history[1]:
            raise RuntimeError("The LLM selected stop before producing a ranking.")
        operations = [list(x) for x in self.re_history[0:1]][0]
        rankings = [list(x) for x in self.re_history[1:2]][0]
        final_ranking = list(rankings[-1])
        original_order = {self._id_key(x): i for i, x in enumerate(self.candidate_id)}
        score = [final_ranking.index(x) for x in self.candidate_id]
        history = [operations, rankings]
        self.re_history = [[], []]
        return final_ranking, history, score

    def rerank(self, data):
        """Run the full iterative process and return (ranking, history, score)."""
        self.update_data(data)
        self.re_history = [[], []]
        return self.backward_rerank()
