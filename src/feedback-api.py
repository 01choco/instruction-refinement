import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from datasets import load_dataset
import os
from openai import OpenAI
import pandas as pd
from vllm import LLM, SamplingParams

# Custom variables 
MODEL_ID = "princeton-nlp/Llama-3-Base-8B-SFT"                      # model id
N_RESPONSES = 2                        # response count per instruction
TEMPERATURE = 0.9
TOP_P = 1.0
MAX_NEW_TOKENS = 512
REPETITION_PENALTY = 1.05
SEEDS_BASE = 42                        # random seed
USE_FP16_BF16 = True                   # fp16 or bfloat16 if possible

# OpenAI Config 
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def generate_response_openai(cfg, prompt: str) -> str:
    resp = client.responses.create(
        model=cfg.gpt_model,
        input=[
            {"role": "system", "content": "The assistant should provide users with accurate, relevant, and up-to-date information, ensuring that the content is positive, interesting, engaging, educational, and helpful."},
            {"role": "user", "content": prompt}
        ],
        max_output_tokens=cfg.gpt_max_tokens,
    )
    return resp.output_text


def refine_instruction_single(cfg, item: dict):
    """단일 item(dict) 기준으로 refine 수행"""
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
    """dataset: [{instruction, responses, ...}, ...]"""
    prompt_count = len(dataset)
    refined_prompts = [None] * prompt_count
    refine_records = [None] * prompt_count

    def _refine_single(idx_item):
        idx, item = idx_item
        feedback, refined_inst = refine_instruction_single(cfg, item)
        return idx, feedback, refined_inst

    with ThreadPoolExecutor(max_workers=cfg.gpt_worker_count) as executor:
        futures = [executor.submit(_refine_single, (i, item)) for i, item in enumerate(dataset)]
        for future in tqdm(as_completed(futures), total=prompt_count, desc="Refining prompts"):
            idx, feedback, refined_instruction = future.result()
            refined_prompts[idx] = refined_instruction
            refine_records[idx] = feedback

    return refined_prompts, refine_records

class ArmoRMPipeline:
    def __init__(self, model_id, device_map="auto", torch_dtype=torch.bfloat16, truncation=True, trust_remote_code=False, max_length=4096):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding_side = "left"

        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, float]:
        """
        messages: OpenAI chat messages to be scored
        Note: no batching since due to length differences, the model will have to pad to the max length which is not efficient
        Returns: a dictionary with the score between 0 and 1
        """
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
            score = output.score.float().item()
        return score

def get_armo(cfg, dataset):
    rm = ArmoRMPipeline(
        "RLHFlow/ArmoRM-Llama3-8B-v0.1",
        device_map={"": "cuda:0"},  # 단일 GPU 고정
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )    
    
    print(f"Processing {len(dataset)} items...")
    loop_cnt = 0
    while True:
        cnt, dataset= loop(cfg, rm, dataset, loop_cnt)
        print(f"Low scores detected: {cnt} items need feedback.")
        loop_cnt += 1
        if cnt == 0 or loop_cnt >= cfg.max_loop:
            with open(f"{cfg.feedback_path}/final_left.jsonl", 'w', encoding='utf-8') as f:
                for data in dataset:
                    json.dump(data, f, ensure_ascii=False)
                    f.write('\n')

            original_path = f"{cfg.feedback_path}/final_original.jsonl"
            refined_path = f"{cfg.feedback_path}/final_refined.jsonl"

            cnt = 0
            for i in range(loop_cnt):
                uf_path = f"{cfg.feedback_path}/ultrafeedback-{i}.jsonl"
                if not os.path.exists(uf_path):
                    continue
                with open(uf_path, 'r', encoding='utf-8') as f:                    
                    for line in f:
                        item = json.loads(line.strip())
                        original_instruction = item['original_instruction'] if 'original_instruction' in item else item['instruction']
                        refined_instruction = item['instruction']
                        original_responses = item['original_responses'] if 'original_responses' in item else item['responses']
                        refined_responses = item['responses']

                        with open(original_path, 'a', encoding='utf-8') as of:
                            json.dump({
                                "instruction": original_instruction,
                                "responses": original_responses
                            }, of, ensure_ascii=False)
                            of.write('\n')
                        with open(refined_path, 'a', encoding='utf-8') as rf:
                            json.dump({
                                "instruction": refined_instruction,
                                "responses": refined_responses
                            }, rf, ensure_ascii=False)
                            rf.write('\n')
                        cnt += 1
                print(f"{i} step data processing complete.")

            print(f"Total {cnt} refined items processed.")

            original_left_path = f"{cfg.feedback_path}/final_original_left.jsonl"
            refined_left_path = f"{cfg.feedback_path}/final_refined_left.jsonl"

            cnt_left = 0
            with open(f"{cfg.feedback_path}/final_left.jsonl", 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    original_instruction = item['original_instruction'] if 'original_instruction' in item else item['instruction']
                    refined_instruction = item['instruction']
                    original_responses = item['original_responses'] if 'original_responses' in item else item['responses']
                    refined_responses = item['responses']

                    with open(original_left_path, 'a', encoding='utf-8') as of:
                        json.dump({
                            "instruction": original_instruction,
                            "responses": original_responses
                        }, of, ensure_ascii=False)
                        of.write('\n')
                    with open(refined_left_path, 'a', encoding='utf-8') as rf:
                        json.dump({
                            "instruction": refined_instruction,
                            "responses": refined_responses
                        }, rf, ensure_ascii=False)
                        rf.write('\n')
                    cnt_left += 1

            print(f"Total {cnt_left} left items processed.")
            break

def refine(dataset: list[dict], loop_cnt: int, cfg):
    """
    dataset: RM 점수 낮아서 refine 대상이 된 item 리스트
    return: 새로 생성된 out_obj 리스트
            out_obj = {
              "original_instruction": ...,
              "original_responses": ...,
              "instruction": refined_instruction,
              "responses": new_responses,
            }
    """
    if not dataset:
        return []

    # 1) OpenAI로 instruction refine
    refined_prompts, feedbacks = generate_feedback_instruction(dataset, cfg)

    # 2) vLLM 초기화
    model_path = cfg.model_id  # cfg에 맞게 필드 이름 조정
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
        n=N_RESPONSES,  # 또는 cfg.n_responses
        stop_token_ids=[tokenizer.eos_token_id],
    )
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)

    new_items = []
    refine_path = f"{cfg.feedback_path}/refine.jsonl"
    with open(refine_path, 'a', encoding='utf-8') as f:
        for i, output in enumerate(outputs):
            item = dataset[i]

            original_instruction = (
                item["instruction"] if loop_cnt == 0 else item["original_instruction"]
            )
            original_responses = (
                item["responses"] if loop_cnt == 0 else item["original_responses"]
            )
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

def loop(cfg, rm, dataset, loop_cnt):
    cnt = 0
    new_dataset = []
    need_refine = []

    refine_path = f"{cfg.feedback_path}/refine-old.jsonl"
    if os.path.exists(refine_path):
        try:
            refine_df = pd.read_json(refine_path, lines=True, encoding='utf-8')
        except ValueError as e:
            print(f"Error reading refine.jsonl: {e}")
            refine_df = pd.DataFrame()
    else:
        refine_df = pd.DataFrame()

    for item in tqdm(dataset, desc="Processing dataset"):
        # set response data
        similarity_scores = []

        prompt = item["instruction"]
        if len(item["responses"]) > N_RESPONSES:
            item["responses"] = item["responses"][:N_RESPONSES]
        
        for j in range(len(item["responses"])):
            generated_response = item["responses"][j]
            score = rm([{"role": "user", "content": prompt}, {"role": "assistant", "content": generated_response}])
            similarity_scores.append(score)

        item['scores'] = similarity_scores

        # Check if the instruction already exists in refine-old.jsonl
        if item["instruction"] in refine_df.get('instruction', []).values:
            matched_row = refine_df[refine_df['instruction'] == item["instruction"]].iloc[0]

            out_obj = {
                "original_instruction": matched_row["original_instruction"],
                "original_responses": matched_row["original_responses"],
                "instruction": matched_row["refined_instruction"],
                "responses": matched_row["refined_responses"],
            }

            with open(f"{cfg.feedback_path}/refine.jsonl", 'a', encoding='utf-8') as f:
                json.dump(matched_row.to_dict(), f, ensure_ascii=False)
                f.write('\n')
            cnt += 1 
            new_dataset.append(out_obj)
        else:
            if cfg.threshold == "max":
                if max(similarity_scores) < cfg.gamma:  # threshold
                    # print(f"Low scores detected: {similarity_scores}")
                    cnt += 1
                    need_refine.append(item)
                else:
                    with open(f"{cfg.feedback_path}/ultrafeedback-{loop_cnt}.jsonl", 'a', encoding='utf-8') as f:
                        json.dump(item, f, ensure_ascii=False)
                        f.write('\n')
            elif cfg.threshold == "mean":
                if sum(similarity_scores) / len(similarity_scores) < cfg.gamma:  # threshold
                    # print(f"Low scores detected: {similarity_scores}")
                    cnt += 1
                    need_refine.append(item)
                else:
                    with open(f"{cfg.feedback_path}/ultrafeedback-{loop_cnt}.jsonl", 'a', encoding='utf-8') as f:
                        json.dump(item, f, ensure_ascii=False)
                        f.write('\n')
            elif cfg.threshold == "min":
                if min(similarity_scores) < cfg.gamma:  # threshold
                    # print(f"Low scores detected: {similarity_scores}")
                    cnt += 1
                    need_refine.append(item)
                else:
                    with open(f"{cfg.feedback_path}/ultrafeedback-{loop_cnt}.jsonl", 'a', encoding='utf-8') as f:
                        json.dump(item, f, ensure_ascii=False)
                        f.write('\n')

    refined = refine(need_refine, loop_cnt, cfg)
    for r in refined:
        new_dataset.append(r)

    return cnt, new_dataset


@hydra.main(version_base=None, config_path="")
def main(cfg: DictConfig):
    print(f"Loaded config name: {cfg}")

    os.makedirs(cfg.feedback_path, exist_ok=True)
    ds = load_dataset("json", data_files=cfg.inference_path, split="train")
    get_armo(cfg, ds)
    print(f"Data Processing completed.")

if __name__ == "__main__":
    main()
