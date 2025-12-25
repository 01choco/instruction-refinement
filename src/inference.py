# run_infer_hf_llama3_jsonl.py
import os
import json
import time
from typing import Dict, Iterable, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from itertools import islice
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
START = 1                            
END = 25706                        

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

    # 총 개수 파악은 선택
    try:
        total = sum(1 for _ in read_jsonl(INPUT_PATH))
    except Exception:
        total = 0

    total1 = START
    total2 = END

    read_idx = total1
    saved = 0
    # it = islice(read_jsonl(INPUT_PATH), read_idx - 1, total2)
    it = read_jsonl(INPUT_PATH)
    
    print("start from : ", total1, " to ", total2)
    prompts = []

    for rec in tqdm(it, total=(total2 - total1 + 1), desc="Processing"):
        if read_idx == total2:
            break
        instr = rec.get("instruction")
        if not instr:
            continue
        if instr in done:
            continue
        prompts.append(instr)

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

    template_prompts = [safe_prompt(build_chat_prompt(p)) for p in prompts]

    sampling_params = SamplingParams(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=max_new,
        n=N_RESPONSES,
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
        saved += 1
        read_idx += 1

    print("done")

if __name__ == "__main__":
    main()
