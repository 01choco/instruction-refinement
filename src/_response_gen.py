import json
import os

import hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

N_RESPONSES = 2

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


@hydra.main(version_base=None, config_path="")
def main(cfg: DictConfig):
    loop_cnt = int(os.environ.get("LOOP_CNT", "0"))

    need_refine_path = f"{cfg.feedback_path}/_tmp_need_refine_{loop_cnt}.jsonl"
    next_base_path = f"{cfg.feedback_path}/_tmp_next_base_{loop_cnt}.jsonl"
    refined_prompts_path = f"{cfg.feedback_path}/_tmp_refined_prompts_{loop_cnt}.jsonl"

    next_path = f"{cfg.feedback_path}/_tmp_next_{loop_cnt}.jsonl"
    refine_log_path = f"{cfg.feedback_path}/refine.jsonl"
    os.makedirs(cfg.feedback_path, exist_ok=True)

    need_refine = load_jsonl(need_refine_path)
    next_base = load_jsonl(next_base_path)
    refined_items = load_jsonl(refined_prompts_path)

    refined_prompts = [it["refined_instruction"] for it in refined_items]
    feedbacks = [it["feedback"] for it in refined_items]

    # 메인 LLM 단독 로드
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

    SYSTEM_PROMPT = ("")

    def build_chat_prompt(user_text: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        # 모델 토크나이저가 chat_template을 갖고 있으면 이게 정석
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    template_prompts = [safe_prompt(build_chat_prompt(p)) for p in refined_prompts]

    sampling_params = SamplingParams(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=max_new,
        n=N_RESPONSES,
        stop=STOPS,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    outputs = llm.generate(template_prompts, sampling_params=sampling_params, use_tqdm=True)

    new_items = []
    with open(refine_log_path, "a", encoding="utf-8") as f:
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

    next_all = next_base + new_items
    save_jsonl(next_all, next_path, mode="w")
    print(f"[LOOP {loop_cnt}] wrote {next_path} (size={len(next_all)})")


if __name__ == "__main__":
    main()
