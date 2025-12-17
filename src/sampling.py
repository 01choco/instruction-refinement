import json
import hydra
from datasets import load_dataset
import random

def sample_data(cfg, train):
    cnt = 0

    evol_instruct = []
    flan_v2_niv2 = []
    flan_v2_cot = []
    flan_v2_p3 = []
    flan_v2_flan2021 = []
    false_qa = []
    sharegpt = []
    truthful_qa = []
    ultrachat = []

    for i in range(len(train)):
        item = train[i]
        
        # set response data
        data_source = item["source"]       

        if data_source == "evol_instruct":
            evol_instruct.append(item)
        elif data_source == "false_qa":
            false_qa.append(item)
        elif data_source == "flan_v2_niv2":
            flan_v2_niv2.append(item)
        elif data_source == "flan_v2_cot":
            flan_v2_cot.append(item)
        elif data_source == "flan_v2_p3":
            flan_v2_p3.append(item)
        elif data_source == "flan_v2_flan2021":
            flan_v2_flan2021.append(item)
        elif data_source == "sharegpt":
            sharegpt.append(item)
        elif data_source == "truthful_qa":
            truthful_qa.append(item)
        elif data_source == "ultrachat":
            ultrachat.append(item)
        else:
            print(f"Warning: Unknown data source {data_source}. Skipping.")
            continue

    new_datas = []

    def sampling (datas, percentage):
        random.seed(cfg.seed)
        sample_size = int(len(datas) * percentage)
        return random.sample(datas, sample_size)
    
    percentage = cfg.split 
    for [datas, name] in zip([evol_instruct, false_qa, flan_v2_niv2, flan_v2_cot, flan_v2_p3, flan_v2_flan2021,sharegpt, truthful_qa, ultrachat], 
                            ["evol_instruct", "false_qa", "flan_v2_niv2", "flan_v2_cot", "flan_v2_p3", "flan_v2_flan2021", "sharegpt", "truthful_qa", "ultrachat"]):
        sampled_datas = sampling(datas, percentage)
        new_datas.extend(sampled_datas)
        print(f"Sampled {len(sampled_datas)} from {name} (Total: {len(datas)})")

    if len(new_datas) > 0:
        with open(cfg.output_path, 'a') as file:
            for d in new_datas:  # 리스트 안 모든 원소 처리
                json.dump(d, file, ensure_ascii=False)  
                file.write('\n')
                cnt += 1
                print(f"1 data appended. {cnt} / {len(new_datas)} Data converted.")
    return cnt 

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
