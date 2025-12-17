import json
import os

import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

N_RESPONSES = 2

_EVAL_LLM = None
_EVAL_TOK = None
_REFINE_LLM = None
_REFINE_TOK = None

def init_eval_llm(cfg):
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

def init_refine_llm(cfg):
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

def eval_generate_batch(cfg, prompts: list[str], show_tqdm: bool = True) -> list[str]:
    init_eval_llm(cfg)
    sp = SamplingParams(
        temperature=cfg.eval_temperature,
        top_p=cfg.eval_top_p,
        max_tokens=cfg.eval_max_tokens,
        n=1,
        stop_token_ids=[_EVAL_TOK.eos_token_id],
    )
    outs = _EVAL_LLM.generate(prompts, sp, use_tqdm=show_tqdm)
    return [o.outputs[0].text.strip() for o in outs]

def refine_generate_batch(cfg, prompts: list[str], show_tqdm: bool = True) -> list[str]:
    init_refine_llm(cfg)
    sp = SamplingParams(
        temperature=cfg.refine_temperature,
        top_p=cfg.refine_top_p,
        max_tokens=cfg.refine_max_tokens,
        n=1,
        stop_token_ids=[_REFINE_TOK.eos_token_id],
    )
    outs = _REFINE_LLM.generate(prompts, sp, use_tqdm=show_tqdm)
    return [o.outputs[0].text.strip() for o in outs]


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


def generate_feedback_instruction(dataset: list[dict], cfg):
    """
    dataset: [{instruction, responses, ...}, ...]
    return: refined_prompts(list[str]), feedbacks(list[str])
    """
    prompt_count = len(dataset)
    original_prompts = [item["instruction"] for item in dataset]

    # (A) evaluator 프롬프트들 만들기
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

    # (A) evaluator 배치 생성 (포맷 정확도가 중요해서 temperature 낮게 추천)
    feedbacks = eval_generate_batch(cfg, eval_prompts, show_tqdm=True)
    
    # (B) refiner 프롬프트들 만들기
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

    # (B) refiner 배치 생성
    refined_prompts = refine_generate_batch(cfg, refine_prompts, show_tqdm=True)

    return refined_prompts, feedbacks

def refine_and_generate(cfg: DictConfig, need_refine: list[dict], loop_cnt: int) -> list[dict]:
    if not need_refine:
        return []

    refined_prompts, feedbacks = generate_feedback_instruction(need_refine, cfg)

    # vLLM init (원본과 동일 필드 사용)
    model_path = cfg.model_id
    max_ctx = cfg.max_input_length
    max_new = cfg.max_new_tokens
    max_prompt_len = max_ctx - max_new

    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        dtype="bfloat16",
        max_model_len=max_ctx,
        load_format="auto",
        seed=42,
        tensor_parallel_size=cfg.tp,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def safe_prompt(p: str) -> str:
        enc = tokenizer(p, add_special_tokens=False)
        ids = enc.input_ids
        if len(ids) <= max_prompt_len:
            return p
        print(f"[WARN] truncating prompt {len(ids)} -> {max_ctx}")
        ids = ids[:max_prompt_len]
        return tokenizer.decode(ids, skip_special_tokens=False)

    prompts = [safe_prompt(p) for p in refined_prompts]

    sampling_params = SamplingParams(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=max_new,
        n=N_RESPONSES,
        stop_token_ids=[tokenizer.eos_token_id],
    )
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)

    new_items = []
    refine_path = f"{cfg.feedback_path}/refine.jsonl"
    os.makedirs(cfg.feedback_path, exist_ok=True)

    with open(refine_path, "a", encoding="utf-8") as f:
        for i, output in enumerate(outputs):
            item = need_refine[i]

            original_instruction = item["instruction"] if loop_cnt == 0 else item["original_instruction"]
            original_responses = item["responses"] if loop_cnt == 0 else item["original_responses"]

            refined_instruction = refined_prompts[i]
            responses = [out.text for out in output.outputs]
            feedback = feedbacks[i]

            out_obj = {
                "original_instruction": original_instruction,
                "original_responses": original_responses,
                "instruction": refined_instruction,
                "responses": responses,
            }
            new_items.append(out_obj)

            refine_obj = {
                "instruction": item["instruction"],
                "responses": item["responses"],
                "feedback": feedback,
                "original_instruction": original_instruction,
                "original_responses": original_responses,
                "refined_instruction": refined_instruction,
                "refined_responses": responses,
                "loop_cnt": loop_cnt,
            }
            json.dump(refine_obj, f, ensure_ascii=False)
            f.write("\n")

    return new_items


@hydra.main(version_base=None, config_path="")
def main(cfg: DictConfig):
    loop_cnt = int(os.environ.get("LOOP_CNT", "0"))

    need_refine_path = f"{cfg.feedback_path}/_tmp_need_refine_{loop_cnt}.jsonl"
    next_base_path = f"{cfg.feedback_path}/_tmp_next_base_{loop_cnt}.jsonl"
    next_path = f"{cfg.feedback_path}/_tmp_next_{loop_cnt}.jsonl"

    need_refine = load_jsonl(need_refine_path)
    next_base = load_jsonl(next_base_path)

    refined_items = refine_and_generate(cfg, need_refine, loop_cnt)

    # 다음 loop dataset = (cache-hit으로 이미 만든 next_base) + (이번 loop에서 새로 refine한 refined_items)
    next_all = next_base + refined_items
    save_jsonl(next_all, next_path, mode="w")

    print(f"[LOOP {loop_cnt}] wrote {next_path} (size={len(next_all)})")


if __name__ == "__main__":
    main()
