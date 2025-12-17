import json
import os
import subprocess

import hydra
from omegaconf import DictConfig


def run_cmd(cmd: list[str], env: dict):
    subprocess.check_call(cmd, env=env)

def finalize_outputs(cfg: DictConfig, final_dataset_path: str, loop_cnt: int):
    dataset = []
    with open(final_dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))

    os.makedirs(cfg.feedback_path, exist_ok=True)

    with open(f"{cfg.feedback_path}/final_left.jsonl", "w", encoding="utf-8") as f:
        for data in dataset:
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")

    original_path = f"{cfg.feedback_path}/final_original.jsonl"
    refined_path = f"{cfg.feedback_path}/final_refined.jsonl"

    original_left_path = f"{cfg.feedback_path}/final_original_left.jsonl"
    refined_left_path = f"{cfg.feedback_path}/final_refined_left.jsonl"

    cnt = 0
    for i in range(loop_cnt):
        uf_path = f"{cfg.feedback_path}/ultrafeedback-{i}.jsonl"
        if not os.path.exists(uf_path):
            continue
        with open(uf_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                original_instruction = item.get("original_instruction", item["instruction"])
                refined_instruction = item["instruction"]
                original_responses = item.get("original_responses", item["responses"])
                refined_responses = item["responses"]

                with open(original_path, "a", encoding="utf-8") as of:
                    json.dump({"instruction": original_instruction, "responses": original_responses}, of, ensure_ascii=False)
                    of.write("\n")
                with open(refined_path, "a", encoding="utf-8") as rf:
                    json.dump({"instruction": refined_instruction, "responses": refined_responses}, rf, ensure_ascii=False)
                    rf.write("\n")
                with open(original_left_path, "a", encoding="utf-8") as of:
                    json.dump({"instruction": original_instruction, "responses": original_responses}, of, ensure_ascii=False)
                    of.write("\n")
                with open(refined_left_path, "a", encoding="utf-8") as rf:
                    json.dump({"instruction": refined_instruction, "responses": refined_responses}, rf, ensure_ascii=False)
                    rf.write("\n")

                cnt += 1


    with open(f"{cfg.feedback_path}/final_left.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            original_instruction = item.get("original_instruction", item["instruction"])
            refined_instruction = item["instruction"]
            original_responses = item.get("original_responses", item["responses"])
            refined_responses = item["responses"]

            with open(original_left_path, "a", encoding="utf-8") as of:
                json.dump({"instruction": original_instruction, "responses": original_responses}, of, ensure_ascii=False)
                of.write("\n")
            with open(refined_left_path, "a", encoding="utf-8") as rf:
                json.dump({"instruction": refined_instruction, "responses": refined_responses}, rf, ensure_ascii=False)
                rf.write("\n")


@hydra.main(version_base=None, config_path="")
def main(cfg: DictConfig):
    os.makedirs(cfg.feedback_path, exist_ok=True)

    cur_path = cfg.inference_path

    env_armo = os.environ.get("ARMO_ENV", "qref_armo")
    env_gen = os.environ.get("GEN_ENV", "qref_gen")

    loop_cnt = 0
    while True:
        scored_path = f"{cfg.feedback_path}/_scored_{loop_cnt}.jsonl"
        next_path = f"{cfg.feedback_path}/_next_{loop_cnt}.jsonl"

        base_env = os.environ.copy()

        # scoring with ArmoRM (armo env)
        env1 = base_env.copy()
        env1["LOOP_CNT"] = str(loop_cnt)
        env1["INPUT_PATH"] = cur_path
        env1["SCORED_PATH"] = scored_path
        print(f"[LOOP {loop_cnt}] scoring start: {cur_path}")
        
        run_cmd(
            ["conda", "run", "-n", env_armo, "python", "src/feedback_armo.py",
             "--config-path", "../config", "--config-name", "config-api"],
            env=env1
        )
        print(f"[LOOP {loop_cnt}] scoring done -> {scored_path}")

        # generation (gen env)
        env2 = base_env.copy()
        env2["LOOP_CNT"] = str(loop_cnt)
        env2["SCORED_PATH"] = scored_path
        env2["NEXT_PATH"] = next_path

        print(f"[LOOP {loop_cnt}] refine/gen start")

        run_cmd(
            ["conda", "run", "-n", env_gen, "python", "-u", "src/feedback_gen.py",
             "--config-path", "../config", "--config-name", "config-api"],
            env=env2
        )

        print(f"[LOOP {loop_cnt}] refine/gen done -> {next_path}")
        
        # 다음 loop 입력 갱신
        cur_path = next_path

        loop_cnt += 1
        if loop_cnt >= cfg.max_loop:
            break

    finalize_outputs(cfg, cur_path, loop_cnt)

if __name__ == "__main__":
    main()
