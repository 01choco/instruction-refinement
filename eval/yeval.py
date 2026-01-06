import json
import re
from typing import Dict
from datasets import load_dataset
from omegaconf import DictConfig
import hydra
import os
import csv
from tqdm import tqdm
import json
from typing import Dict, List
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import sys
from datasets import load_dataset


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
    rm = ArmoRMPipeline("RLHFlow/ArmoRM-Llama3-8B-v0.1", trust_remote_code=True)
    print(f"Processing {len(dataset)} items...")

    avg_scores = 0

    for item in tqdm(dataset, desc="Processing dataset"):
        # set response data
        scores = []

        prompt = item["instruction"]
        for j in range(len(item["responses"])):
            generated_response = item["responses"][j]
            score = rm([{"role": "user", "content": prompt}, {"role": "assistant", "content": generated_response}])
            scores.append(score)

        item['scores'] = scores
        item['avg_score'] = sum(scores) / len(scores) if scores else 0
        avg_scores += sum(scores)

        with open(f"{cfg.armorm_path}", 'a', encoding='utf-8') as f:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    avg_scores /= (len(dataset)*2) if dataset else 1

    print(f"Average Score: {avg_scores}")

@hydra.main(version_base=None, config_path="")
def main(cfg: DictConfig):
    print(f"Loaded config name: {cfg}")

    ds = load_dataset("json", data_files=cfg.refined_path, split="train")

    if cfg.sampling and cfg.size > 0:
        sample_size = cfg.size
        ds = ds.shuffle(seed=cfg.seed).select(range(sample_size))
        print(f"Randomly sampled {sample_size} items from the dataset.")
    get_armo(cfg, ds)

if __name__ == "__main__":
    main()
