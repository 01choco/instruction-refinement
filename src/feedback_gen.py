import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import hydra
from omegaconf import DictConfig
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

N_RESPONSES = 2
REPETITION_PENALTY = 1.05

STOPS = [
    # user role
    "\n<|start_header_id|>user<|end_header_id|>",
    "<|start_header_id|>user<|end_header_id|>\n\n",
    "\nuser", "User:", "user\n\n", "<|user|>",
]

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


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


def generate_response_openai(cfg, prompt: str) -> str:
    resp = client.responses.create(
        model=cfg.gpt_model,
        input=[
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ],
        max_output_tokens=cfg.gpt_max_tokens,
    )
    return resp.output_text


def refine_instruction_single(cfg, item: dict):
    prompt = item["instruction"]

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
    feedback = generate_response_openai(cfg, evaluation_template)

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
    refined_instruction = generate_response_openai(cfg, refinement)
    return feedback, refined_instruction


def generate_feedback_instruction(dataset: list[dict], cfg):
    prompt_count = len(dataset)
    refined_prompts = [None] * prompt_count
    refine_records = [None] * prompt_count

    def _refine_single(idx_item):
        idx, item = idx_item
        feedback, refined_inst = refine_instruction_single(cfg, item)
        return idx, feedback, refined_inst

    with ThreadPoolExecutor(max_workers=cfg.gpt_worker_count) as executor:
        futures = [executor.submit(_refine_single, (i, item)) for i, item in enumerate(dataset)]
        for future in tqdm(as_completed(futures), total=prompt_count, desc="Refining prompts", dynamic_ncols=True):
            idx, feedback, refined_instruction = future.result()
            refined_prompts[idx] = refined_instruction
            refine_records[idx] = feedback

    return refined_prompts, refine_records


def refine_and_generate(cfg: DictConfig, need_refine: list[dict], loop_cnt: int) -> list[dict]:
    if not need_refine:
        return []

    refined_prompts, feedbacks = generate_feedback_instruction(need_refine, cfg)

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
        return tokenizer.decode(ids, skip_special_tokens=True)

    SYSTEM_PROMPT = ("")

    def build_chat_prompt(user_text: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    template_prompts = [safe_prompt(build_chat_prompt(p)) for p in refined_prompts]
    
    sampling_params = SamplingParams(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=50,
        max_tokens=max_new,
        n=N_RESPONSES,
        stop=STOPS,
        repetition_penalty=REPETITION_PENALTY,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    outputs = llm.generate(template_prompts, sampling_params=sampling_params, use_tqdm=True)

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

    next_all = next_base + refined_items
    save_jsonl(next_all, next_path, mode="w")

    print(f"[LOOP {loop_cnt}] wrote {next_path} (size={len(next_all)})")


if __name__ == "__main__":
    main()
