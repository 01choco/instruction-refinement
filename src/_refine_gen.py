import json
import os

import hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


_REFINE_LLM = None
_REFINE_TOK = None

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

def init_refine_llm(cfg: DictConfig):
    global _REFINE_LLM, _REFINE_TOK
    if _REFINE_LLM is not None:
        return
    _REFINE_LLM = LLM(
        model=cfg.refine_model_id,
        tokenizer=cfg.refine_model_id,
        dtype=cfg.refine_dtype,
        max_model_len=cfg.refine_max_input_length,
        tensor_parallel_size=cfg.refine_tp,
        seed=cfg.seed,
        load_format="auto",
    )
    _REFINE_TOK = AutoTokenizer.from_pretrained(cfg.refine_model_id, use_fast=True)


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


def refine_generate_batch(cfg: DictConfig, prompts: list[str]) -> list[str]:
    init_refine_llm(cfg)

    model_path = cfg.refine_model_id
    max_ctx = cfg.refine_max_input_length
    max_new = cfg.refine_max_tokens
    max_prompt_len = max_ctx - max_new

    SYSTEM_PROMPT = ("")

    def safe_prompt(p: str) -> str:
        enc = _REFINE_TOK(p, add_special_tokens=False)
        ids = enc.input_ids
        if len(ids) <= max_prompt_len:
            return p
        print(f"[WARN] truncating prompt {len(ids)} -> {max_ctx}")
        ids = ids[:max_prompt_len]
        return _REFINE_TOK.decode(ids, skip_special_tokens=True)


    def build_chat_prompt(user_text: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        # 모델 토크나이저가 chat_template을 갖고 있으면 이게 정석
        return _REFINE_TOK.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    template_prompts = [safe_prompt(build_chat_prompt(p)) for p in prompts]

    sp = SamplingParams(
        temperature=cfg.refine_temperature,
        top_p=cfg.refine_top_p,
        max_tokens=cfg.refine_max_tokens,
        n=1,
        stop=STOPS,
        stop_token_ids=[_REFINE_TOK.eos_token_id],
    )
    outs = _REFINE_LLM.generate(template_prompts, sp, use_tqdm=True)
    return [o.outputs[0].text.strip() for o in outs]


def build_refine_prompts(original_prompts: list[str], feedbacks: list[str]) -> list[str]:
    refine_prompts = []
    for prompt, feedback in zip(original_prompts, feedbacks):
        refinement = f"""You are an instruction refiner. 
Given an Original Instruction and its Evaluation, rewrite the instruction so it is clear, specific, complete, safe, feasible, concise, and consistent.

STRICT RULES:
1. Output ONLY the refined instruction text — no labels, no commentary, no examples.
2. The refined instruction must be executable directly by a model (not guidelines).
3. Preserve the intent of the original, but resolve ambiguity using the feedback.
4. Do not include the words "Refined Instruction" in the output.
5. Write a single instruction only.

Original Instruction:
{prompt}

Evaluation:
{feedback}
"""
        refine_prompts.append(refinement)
    return refine_prompts


@hydra.main(version_base=None, config_path="")
def main(cfg: DictConfig):
    loop_cnt = int(os.environ.get("LOOP_CNT", "0"))

    need_refine_path = f"{cfg.feedback_path}/_tmp_need_refine_{loop_cnt}.jsonl"
    feedback_path = f"{cfg.feedback_path}/_tmp_feedback_{loop_cnt}.jsonl"
    out_path = f"{cfg.feedback_path}/_tmp_refined_prompts_{loop_cnt}.jsonl"

    need_refine = load_jsonl(need_refine_path)
    fb_items = load_jsonl(feedback_path)

    # instruction 순서가 동일하다는 전제(01 단계가 need_refine 순서대로 저장)
    original_prompts = [it["instruction"] for it in need_refine]
    feedbacks = [it["feedback"] for it in fb_items]

    refine_prompts = build_refine_prompts(original_prompts, feedbacks)
    refined_prompts = refine_generate_batch(cfg, refine_prompts)

    out_items = []
    for orig, fb, refined in zip(original_prompts, feedbacks, refined_prompts):
        out_items.append({
            "instruction": orig,
            "feedback": fb,
            "refined_instruction": refined,
        })

    save_jsonl(out_items, out_path, mode="w")
    print(f"[LOOP {loop_cnt}] wrote {out_path} (size={len(out_items)})")


if __name__ == "__main__":
    main()
