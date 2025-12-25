import json
import os

import hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


_REFINE_LLM = None
_REFINE_TOK = None


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
    sp = SamplingParams(
        temperature=cfg.refine_temperature,
        top_p=cfg.refine_top_p,
        max_tokens=cfg.refine_max_tokens,
        n=1,
        stop_token_ids=[_REFINE_TOK.eos_token_id],
    )
    outs = _REFINE_LLM.generate(prompts, sp, use_tqdm=True)
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
