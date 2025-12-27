import json
import os

import hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


_EVAL_LLM = None
_EVAL_TOK = None

STOPS = [
    # user role
    "\n<|start_header_id|>user<|end_header_id|>",
    "<|start_header_id|>user<|end_header_id|>\n\n",
    "\nuser", "User:", "user\n\n", "<|user|>",

    # assistant role
    "\n<|start_header_id|>assistant<|end_header_id|>",
    "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "\nassistant", "Assistant:", "assistant\n\n", "<|assistant|>"
]


def init_eval_llm(cfg: DictConfig):
    global _EVAL_LLM, _EVAL_TOK
    if _EVAL_LLM is not None:
        return
    _EVAL_LLM = LLM(
        model=cfg.eval_model_id,
        tokenizer=cfg.eval_model_id,
        dtype=cfg.eval_dtype,
        max_model_len=cfg.eval_max_input_length,
        tensor_parallel_size=cfg.eval_tp,
        seed=cfg.seed,
        load_format="auto",
    )
    _EVAL_TOK = AutoTokenizer.from_pretrained(cfg.eval_model_id, use_fast=True)


def load_jsonl(path: str) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def save_jsonl(items: list[dict], path: str, mode: str = "w"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for it in items:
            json.dump(it, f, ensure_ascii=False)
            f.write("\n")


def build_eval_prompts(original_prompts: list[str]) -> list[str]:
    eval_prompts = []
    for prompt in original_prompts:
        evaluation_template = f"""You are an evaluator. Given an Original Instruction,
evaluate the instruction using the criteria below. 
Follow these STRICT rules:
1. Output must start with exactly 'Evaluation:' on its own line.
2. You must include ALL 7 criteria in the following order: 
   Clarity, Specificity, Completeness, Safety, Answerability, Conciseness, FormatConsistency.
3. Each line must follow the format:
   * <Criterion>: <digit 1-5>/5 - <one concise note>
4. Do NOT add any text before or after the evaluation block.

Output format:
Evaluation:
* Clarity: <1-5>/5 - <one-line note>
* Specificity: <1-5>/5 - <one-line note>
* Completeness: <1-5>/5 - <one-line note>
* Safety: <1-5>/5 - <one-line note>
* Answerability: <1-5>/5 - <one-line note>
* Conciseness: <1-5>/5 - <one-line note>
* FormatConsistency: <1-5>/5 - <one-line note>

---

### Few-shot Examples

Original Instruction:
Write something about animals

Evaluation:
* Clarity: 2/5 - vague request
* Specificity: 2/5 - no length, no scope
* Completeness: 2/5 - missing output format
* Safety: 5/5 - safe
* Answerability: 4/5 - feasible but broad
* Conciseness: 3/5 - some redundancy
* FormatConsistency: 3/5 - unspecified output format

###

Original Instruction:
Make a paragraph using that language

Evaluation:
* Clarity: 2/5 - unclear what "that language" refers to
* Specificity: 2/5 - format is defined (paragraph) but content is vague
* Completeness: 2/5 - missing target language or subject
* Safety: 5/5 - safe request
* Answerability: 2/5 - partially answerable but underspecified
* Conciseness: 4/5 - concise but incomplete
* FormatConsistency: 3/5 - loosely consistent but ambiguous wording

###

Original Instruction:
{prompt}
"""
        eval_prompts.append(evaluation_template)
    return eval_prompts


def eval_generate_batch(cfg: DictConfig, prompts: list[str]) -> list[str]:
    init_eval_llm(cfg)

    model_path = cfg.eval_model_id
    max_ctx = cfg.eval_max_input_length
    max_new = cfg.eval_max_tokens
    max_prompt_len = max_ctx - max_new

    SYSTEM_PROMPT = ("")

    def safe_prompt(p: str) -> str:
        enc = _EVAL_TOK(p, add_special_tokens=False)
        ids = enc.input_ids
        if len(ids) <= max_prompt_len:
            return p
        print(f"[WARN] truncating prompt {len(ids)} -> {max_ctx}")
        ids = ids[:max_prompt_len]
        return _EVAL_TOK.decode(ids, skip_special_tokens=True)
    
    def build_chat_prompt(user_text: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        # 모델 토크나이저가 chat_template을 갖고 있으면 이게 정석
        return _EVAL_TOK.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    
    template_prompts = [safe_prompt(build_chat_prompt(p)) for p in prompts]

    sp = SamplingParams(
        temperature=cfg.eval_temperature,
        top_p=cfg.eval_top_p,
        max_tokens=cfg.eval_max_tokens,
        n=1,
        stop=STOPS,
        stop_token_ids=[_EVAL_TOK.eos_token_id],
    )
    outs = _EVAL_LLM.generate(template_prompts, sp, use_tqdm=True)
    return [o.outputs[0].text.strip() for o in outs]

@hydra.main(version_base=None, config_path="")
def main(cfg: DictConfig):
    loop_cnt = int(os.environ.get("LOOP_CNT", "0"))
    need_refine_path = f"{cfg.feedback_path}/_tmp_need_refine_{loop_cnt}.jsonl"
    out_path = f"{cfg.feedback_path}/_tmp_feedback_{loop_cnt}.jsonl"

    dataset = load_jsonl(need_refine_path)
    original_prompts = [item["instruction"] for item in dataset]

    eval_prompts = build_eval_prompts(original_prompts)
    feedbacks = eval_generate_batch(cfg, eval_prompts)

    out_items = []
    for item, feedback in zip(dataset, feedbacks):
        out_items.append({
            "instruction": item["instruction"],
            "feedback": feedback,
        })

    save_jsonl(out_items, out_path, mode="w")
    print(f"[LOOP {loop_cnt}] wrote {out_path} (size={len(out_items)})")


if __name__ == "__main__":
    main()