# run_infer_hf_llama3_jsonl.py
import os
import json
from typing import Dict

from transformers import AutoTokenizer
from tqdm import tqdm

import hydra
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# ======  config  ======

N_RESPONSES = 2                        # response count per instruction
TEMPERATURE = 0.9
TOP_P = 1.0
MAX_NEW_TOKENS = 512
REPETITION_PENALTY = 1.05
SEEDS_BASE = 42                        # random seed
USE_FP16_BF16 = True                   # fp16 or bfloat16 if possible               

STOPS = [
    "\n<|start_header_id|>user<|end_header_id|>",
    "<|start_header_id|>user<|end_header_id|>\n\n",
    "\nuser", "User:", "<|user|>", "user\n\n"
]

# =======================================================================

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def append_jsonl(path: str, obj: Dict):
    with open(path, "a", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
        f.write("\n")

def load_done_instructions(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                instr = rec.get("instruction")
                if instr is not None:
                    done.add(instr)
            except Exception:
                continue
    return done

@hydra.main(version_base=None, config_path="")
def main(cfg):
    OUTPUT_PATH = cfg.inference_path
    INPUT_PATH = cfg.instruction_path

    done = load_done_instructions(OUTPUT_PATH)
    it = read_jsonl(INPUT_PATH)
    prompts = []

    for rec in tqdm(it, desc="Processing"):
        instr = rec.get("instruction")
        if not instr:
            continue
        if instr in done:
            continue
        prompts.append(instr)

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

    SYSTEM_PROMPT = ("You are a helpful, concise assistant")

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

    template_prompts = [safe_prompt(build_chat_prompt(p)) for p in prompts]

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

    for i, output in enumerate(outputs):

        instruction = prompts[i]
        responses = [out.text for out in output.outputs]

        out_obj = {
            "instruction": instruction,
            "responses": responses,
        }
        new_items.append(out_obj)

        append_jsonl(OUTPUT_PATH, out_obj)

    print("done")

if __name__ == "__main__":
    main()
