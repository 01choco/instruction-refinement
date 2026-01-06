import json
import os
from typing import Dict, List

import hydra
import torch
import pandas as pd
from datasets import load_dataset
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

N_RESPONSES = 2

class ArmoRMPipeline:
    def __init__(
        self,
        model_id: str,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        truncation=True,
        trust_remote_code=False,
        max_length=4096,
    ):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding_side = "left"

        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def __call__(self, messages: List[Dict[str, str]]) -> float:
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


def load_jsonl(path: str) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def save_jsonl(items: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            json.dump(it, f, ensure_ascii=False)
            f.write("\n")


@hydra.main(version_base=None, config_path="")
def main(cfg: DictConfig):
    input_path = os.environ["INPUT_PATH"]
    loop_cnt = int(os.environ.get("LOOP_CNT", "0"))

    os.makedirs(cfg.feedback_path, exist_ok=True)

    refine_old_path = f"{cfg.feedback_path}/refine-old-no.jsonl"
    if os.path.exists(refine_old_path):
        try:
            refine_df = pd.read_json(refine_old_path, lines=True, encoding="utf-8")
        except ValueError as e:
            print(f"Error reading refine-old.jsonl: {e}")
            refine_df = pd.DataFrame()
    else:
        refine_df = pd.DataFrame()

    rm = ArmoRMPipeline(
        "RLHFlow/ArmoRM-Llama3-8B-v0.1",
        device_map={"": "cuda:0"},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    ds = load_dataset("json", data_files=input_path, split="train")

    cnt = 0
    next_dataset = []
    need_refine = []

    uf_path = f"{cfg.feedback_path}/ultrafeedback-{loop_cnt}.jsonl"
    refine_path = f"{cfg.feedback_path}/refine.jsonl"

    for item in tqdm(ds, desc=f"[LOOP {loop_cnt}] Processing dataset", dynamic_ncols=True):
        item = dict(item)

        if len(item["responses"]) > N_RESPONSES:
            item["responses"] = item["responses"][:N_RESPONSES]

        prompt = item["instruction"]

        similarity_scores = []
        for resp in item["responses"]:
            s = rm([{"role": "user", "content": prompt}, {"role": "assistant", "content": resp}])
            similarity_scores.append(s)
        item["scores"] = similarity_scores

        if not refine_df.empty and "instruction" in refine_df.columns:
            if refine_df["instruction"].isin([item["instruction"]]).any():
                matched_row = refine_df[refine_df["instruction"] == item["instruction"]].iloc[0]

            out_obj = {
                "original_instruction": matched_row["original_instruction"],
                "original_responses": matched_row["original_responses"],
                "instruction": matched_row["refined_instruction"],
                "responses": matched_row["refined_responses"],
            }

            with open(refine_path, "a", encoding="utf-8") as f:
                json.dump(matched_row.to_dict(), f, ensure_ascii=False)
                f.write("\n")

            cnt += 1
            next_dataset.append(out_obj)
            continue

        below = False
        if cfg.threshold == "max":
            below = max(similarity_scores) < cfg.gamma
        elif cfg.threshold == "mean":
            below = (sum(similarity_scores) / len(similarity_scores)) < cfg.gamma
        else:  # min
            below = min(similarity_scores) < cfg.gamma

        if below:
            cnt += 1
            need_refine.append(item)
        else:
            with open(uf_path, "a", encoding="utf-8") as f:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")

    need_refine_path = f"{cfg.feedback_path}/_tmp_need_refine_{loop_cnt}.jsonl"
    next_base_path = f"{cfg.feedback_path}/_tmp_next_base_{loop_cnt}.jsonl"
    cnt_path = f"{cfg.feedback_path}/_tmp_cnt_{loop_cnt}.txt"

    save_jsonl(need_refine, need_refine_path)
    save_jsonl(next_dataset, next_base_path)
    with open(cnt_path, "w", encoding="utf-8") as f:
        f.write(str(cnt))

    print(f"[LOOP {loop_cnt}] low/cached count = {cnt}")
    print(f"[LOOP {loop_cnt}] wrote {need_refine_path}")
    print(f"[LOOP {loop_cnt}] wrote {next_base_path}")
    print(f"[LOOP {loop_cnt}] wrote {cnt_path}")


if __name__ == "__main__":
    main()
