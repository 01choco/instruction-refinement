import json
import hydra
from datasets import load_dataset
import random


def sampling(datas, percentage):
        n = len(datas)
        sample_size = int(n * percentage)
        return [datas[i] for i in random.sample(range(n), sample_size)]

def sample_data(cfg, datas):
    percentage = cfg.split
    sampled = sampling(datas, percentage)

    cnt = 0
    if len(sampled) > 0:
        with open(cfg.output_path, 'a') as file:
            for d in sampled:  # 리스트 안 모든 원소 처리
                json.dump(d, file, ensure_ascii=False)  
                file.write('\n')
                cnt += 1
                print(f"1 data appended. {cnt} / {len(sampled)} Data converted.")
    return len(sampled)


import hydra

from omegaconf import DictConfig
import sys

@hydra.main(version_base=None, config_path=".")
def main(cfg: DictConfig):
    print(f"Loaded config name: {cfg}")

    data = load_dataset(cfg.data_path, split="train")
    print("-----------------------------")
    print("Data Sampling Started.")

    cnt = sample_data(cfg, data)
    print(f"Data Processing completed. {cnt} data converted and appended.")

if __name__ == "__main__":
    main()
