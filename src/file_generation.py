import json
import os

import hydra
from omegaconf import DictConfig


def load_jsonl(path: str) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


@hydra.main(version_base=None, config_path="")
def main(cfg: DictConfig):
    loop_cnt = int(os.environ["LOOP_DONE"])  # 실제로 돈 loop 횟수
    final_dataset_path = os.environ["FINAL_DATASET_PATH"]  # 마지막 _tmp_next_{k}.jsonl

    os.makedirs(cfg.feedback_path, exist_ok=True)

    dataset = load_jsonl(final_dataset_path)

    # final_left.jsonl
    with open(f"{cfg.feedback_path}/final_left.jsonl", "w", encoding="utf-8") as f:
        for data in dataset:
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")

    original_path = f"{cfg.feedback_path}/final_original.jsonl"
    refined_path = f"{cfg.feedback_path}/final_refined.jsonl"

    cnt = 0
    for i in range(loop_cnt):
        uf_path = f"{cfg.feedback_path}/ultrafeedback-{i}.jsonl"
        if not os.path.exists(uf_path):
            continue
        with open(uf_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                original_instruction = item["original_instruction"] if "original_instruction" in item else item["instruction"]
                refined_instruction = item["instruction"]
                original_responses = item["original_responses"] if "original_responses" in item else item["responses"]
                refined_responses = item["responses"]

                with open(original_path, "a", encoding="utf-8") as of:
                    json.dump({"instruction": original_instruction, "responses": original_responses}, of, ensure_ascii=False)
                    of.write("\n")
                with open(refined_path, "a", encoding="utf-8") as rf:
                    json.dump({"instruction": refined_instruction, "responses": refined_responses}, rf, ensure_ascii=False)
                    rf.write("\n")
                cnt += 1
        print(f"{i} step data processing complete.")

    print(f"Total {cnt} refined items processed.")

    original_left_path = f"{cfg.feedback_path}/final_original_left.jsonl"
    refined_left_path = f"{cfg.feedback_path}/final_refined_left.jsonl"

    with open(f"{cfg.feedback_path}/final_left.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            original_instruction = item["original_instruction"] if "original_instruction" in item else item["instruction"]
            refined_instruction = item["instruction"]
            original_responses = item["original_responses"] if "original_responses" in item else item["responses"]
            refined_responses = item["responses"]

            with open(original_left_path, "a", encoding="utf-8") as of:
                json.dump({"instruction": original_instruction, "responses": original_responses}, of, ensure_ascii=False)
                of.write("\n")
            with open(refined_left_path, "a", encoding="utf-8") as rf:
                json.dump({"instruction": refined_instruction, "responses": refined_responses}, rf, ensure_ascii=False)
                rf.write("\n")

    print(f"Total {len(dataset)} left items processed.")


if __name__ == "__main__":
    main()
