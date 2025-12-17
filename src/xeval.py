import json
import re
from typing import Dict
from datasets import load_dataset
from omegaconf import DictConfig
import hydra
from openai import OpenAI
import os
import csv
from tqdm import tqdm

# OpenAI API 키 설정
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

# 상단 공용 영역
CRITERIA = [
    "Clarity", "Specificity", "Completeness",
    "Safety", "Answerability", "Conciseness",
    "FormatConsistency",
]

EVAL_LINE_RE = re.compile(
    r"^\*\s*(?P<criterion>Clarity|Specificity|Completeness|Safety|Answerability|Conciseness|FormatConsistency)\s*:\s*(?P<score>[1-5])\/5\s*-\s*(?P<note>.+)$"
)

def is_valid_evaluation_block(text: str) -> bool:
    """
    출력이 정확히 요구 형식인지 검사:
    - 'Evaluation:'으로 시작
    - 7개 라인 모두 존재, 각 라인이 정규식에 정확히 매칭
    """
    if not text:
        return False
    lines = [ln.strip() for ln in text.strip().splitlines()]
    # 'Evaluation:' 단독 라인 찾기
    try:
        start_idx = lines.index("Evaluation:")
    except ValueError:
        return False

    body = [ln for ln in lines[start_idx+1:] if ln]  # 빈 줄 제거
    matched = []
    for ln in body:
        m = EVAL_LINE_RE.match(ln)
        if m:
            matched.append(m.group("criterion"))

    # 정확히 7개 기준이 모두 나와야 함
    return sorted(set(matched)) == sorted(CRITERIA) and len(matched) == 7

def parse_evaluation_to_json(evaluation_text: str) -> dict:
    """
    Args:
        evaluation_text: "Evaluation:"

    Returns:
        A dictionary containing the parsed evaluation data.
        Returns an empty dictionary if the input is invalid.
    """
    if not is_valid_evaluation_block(evaluation_text):
        return None
    
    parsed_data = {}
    # 패턴: * <Criterion>: <score>/5 - <note>
    pattern = re.compile(r"\*\s*(?P<criterion>\w+):\s*(?P<score>\d)/5\s*-\s*(?P<note>.+)")

    lines = evaluation_text.strip().split('\n')

    # "Evaluation:" 라인 이후부터 파싱 시작
    start_parsing = False
    for line in lines:
        if line.strip() == "Evaluation:":
            start_parsing = True
            continue
        
        if not start_parsing:
            continue

        match = pattern.match(line.strip())
        if match:
            data = match.groupdict()
            criterion = data['criterion']
            score = int(data['score'])
            note = data['note'].strip()
            
            parsed_data[criterion] = {
                "score": score,
                "note": note
            }
            
    return parsed_data

def generate_response_openai(cfg, prompt):
    resp = client.responses.create(
        model=cfg.gpt_model,  # "gpt-5" 등
        input=[
            {"role": "system", "content": "The assistant should provide users with accurate, relevant, and up-to-date information, ensuring that the content is positive, interesting, engaging, educational, and helpful."},
            {"role": "user", "content": prompt}
        ],
        max_output_tokens=cfg.gpt_max_tokens,
    )
    return resp.output_text


def feedback_instruction(cfg, prompt):
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

    # print(json.dumps(original_json, indent=2, ensure_ascii=False))
    return feedback

def xeval (cfg, dataset):
    output_file = cfg.output_file
    refined_feedback_file = cfg.refined_feedback_file
    original_feedback_file = cfg.original_feedback_file 
    csv_file = cfg.csv_output_file

    cnt = len(dataset)

    Clarity = 0
    Specificity = 0
    Completeness = 0
    Safety = 0
    Answerability = 0
    Conciseness = 0
    FormatConsistency = 0

    refined_Clarity = 0
    refined_Specificity = 0
    refined_Completeness = 0
    refined_Safety = 0
    refined_Answerability = 0
    refined_Conciseness = 0
    refined_FormatConsistency = 0

    avg_win_count = 0
    win_rate = 0

    for item in tqdm(dataset, desc="Processing dataset"):
        win_count = 0

        original_eval = item["feedback"]
        if original_eval == "" or original_eval is None:
            print("Original evaluation missing. Skipping item.")
            cnt -= 1
            continue
        original_json = parse_evaluation_to_json(original_eval)

        if original_json is None:
            for i in range(cfg.max_retries):
                print(f"Retrying original instruction evaluation ({i+1}/{cfg.max_retries})...")
                original_eval = feedback_instruction(cfg, item["instruction"])
                original_json = parse_evaluation_to_json(original_eval)
                if original_json is not None:
                    break
        
        if original_json is None:
            print("Failed to parse original instruction evaluation after retries. Skipping item.")
            original_json = {
            "Clarity": {"score": 0, "note": ""},
            "Specificity": {"score": 0, "note": ""},
            "Completeness": {"score": 0, "note": ""},
            "Safety": {"score": 0, "note": ""},
            "Answerability": {"score": 0, "note": ""},
            "Conciseness": {"score": 0, "note": ""},
            "FormatConsistency": {"score": 0, "note": ""}
            }
        else:
            Clarity += original_json["Clarity"]["score"]
            Specificity += original_json["Specificity"]["score"]
            Completeness += original_json["Completeness"]["score"]
            Safety += original_json["Safety"]["score"]
            Answerability += original_json["Answerability"]["score"]
            Conciseness += original_json["Conciseness"]["score"]
            FormatConsistency += original_json["FormatConsistency"]["score"]

        refined_eval = feedback_instruction(cfg, item["refined_instruction"])
        refined_json = parse_evaluation_to_json(refined_eval)
        if refined_json is None:
            for i in range(cfg.max_retries):
                print(f"Retrying refined instruction evaluation ({i+1}/{cfg.max_retries})...")
                refined_eval = feedback_instruction(cfg, item["refined_instruction"])
                refined_json = parse_evaluation_to_json(refined_eval)
                if refined_json is not None:
                    break
    
        if refined_json is None:
            print("Failed to parse refined instruction evaluation after retries. Skipping item.")
            refined_json = {
            "Clarity": {"score": 0, "note": ""},
            "Specificity": {"score": 0, "note": ""},
            "Completeness": {"score": 0, "note": ""},
            "Safety": {"score": 0, "note": ""},
            "Answerability": {"score": 0, "note": ""},
            "Conciseness": {"score": 0, "note": ""},
            "FormatConsistency": {"score": 0, "note": ""}
            }
        else:
            refined_Clarity += refined_json["Clarity"]["score"]
            refined_Specificity += refined_json["Specificity"]["score"]
            refined_Completeness += refined_json["Completeness"]["score"]
            refined_Safety += refined_json["Safety"]["score"]
            refined_Answerability += refined_json["Answerability"]["score"]
            refined_Conciseness += refined_json["Conciseness"]["score"]
            refined_FormatConsistency += refined_json["FormatConsistency"]["score"]

        if not any(value["score"] > 0 for value in original_json.values()) or not any(value["score"] > 0 for value in refined_json.values()):
            cnt -= 1
            continue
        else:
            for criterion in ["Clarity", "Specificity", "Completeness", "Safety", "Answerability", "Conciseness", "FormatConsistency"]:
                if refined_json[criterion]["score"] >= original_json[criterion]["score"]:
                    win_count += 1

            avg_win_count += win_count

            new_item = item.copy()

            new_item["refined_feedback"] = refined_eval
            new_item["win_count"] = win_count
            new_item["is_refined_better"] = win_count >= 4  # 7개 기준 중 4개 이상이 향상되면 True
            if new_item["is_refined_better"]:
                win_rate += 1

            with open(output_file, "a", encoding="utf-8") as f:
                json.dump(new_item, f, ensure_ascii=False)
                f.write('\n')

            with open(refined_feedback_file, "a", encoding="utf-8") as refined_f, open(original_feedback_file, "a", encoding="utf-8") as original_f:
                json.dump(refined_json, refined_f, ensure_ascii=False)
                refined_f.write('\n')
                json.dump(original_json, original_f, ensure_ascii=False)
                original_f.write('\n')

    avg_win_count /= cnt if dataset else 1
    win_rate /= cnt if dataset else 1

    original_scores = {
        "Clarity": Clarity / cnt if dataset else 0,
        "Specificity": Specificity / cnt if dataset else 0,
        "Completeness": Completeness / cnt if dataset else 0,
        "Safety": Safety / cnt if dataset else 0,
        "Answerability": Answerability / cnt if dataset else 0,
        "Conciseness": Conciseness / cnt if dataset else 0,
        "FormatConsistency": FormatConsistency / cnt if dataset else 0,
    }

    refined_scores = {
        "Clarity": refined_Clarity / cnt if dataset else 0,
        "Specificity": refined_Specificity / cnt if dataset else 0,
        "Completeness": refined_Completeness / cnt if dataset else 0,
        "Safety": refined_Safety / cnt if dataset else 0,
        "Answerability": refined_Answerability / cnt if dataset else 0,
        "Conciseness": refined_Conciseness / cnt if dataset else 0,
        "FormatConsistency": refined_FormatConsistency / cnt if dataset else 0,
    }

    print(f"Average win count: {avg_win_count}")
    print(f"Win rate: {win_rate}")

    print("Original Instructions Average Scores:")
    for criterion, score in original_scores.items():
        print(f"{criterion}: {score}")

    print("Refined Instructions Average Scores:")
    for criterion, score in refined_scores.items():
        print(f"{criterion}: {score}")

    # Save results to csv
    with open(csv_file, mode="a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Original Average", "Refined Average"])
        for criterion in original_scores.keys():
            writer.writerow([criterion, original_scores[criterion], refined_scores[criterion]])
        writer.writerow(["Average Win Count", avg_win_count, ""])
        writer.writerow(["Win Rate", win_rate, ""])
    print(f"Results saved to {csv_file}")

@hydra.main(version_base=None, config_path="")
def main(cfg: DictConfig):
    print(f"Loaded config name: {cfg}")

    ds = load_dataset("json", data_files=cfg.refined_path, split="train")

    if cfg.sampling and cfg.size > 0:
        sample_size = cfg.size
        ds = ds.shuffle(seed=cfg.seed).select(range(sample_size))
        print(f"Randomly sampled {sample_size} items from the dataset.")
    xeval(cfg, ds)

if __name__ == "__main__":
    main()
