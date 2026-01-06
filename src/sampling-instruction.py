import json
import hydra
from omegaconf import DictConfig
import sys

def instruction_data(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile, \
        open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if line.strip():
                data = json.loads(line)
                filtered = {"instruction": data.get("instruction")}
                json.dump(filtered, outfile, ensure_ascii=False)
                outfile.write("\n")

@hydra.main(version_base=None, config_path=".")
def main(cfg: DictConfig):
    print(f"Loaded config name: {cfg}")
    
    cnt = instruction_data(cfg.output_path, cfg.instruction_path)
    print(f"Data Processing completed. {cnt} data converted and appended.")

if __name__ == "__main__":
    main()