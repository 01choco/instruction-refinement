import os
import json
import re
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any, Tuple, List

import hydra
from omegaconf import DictConfig, OmegaConf
from openai import OpenAI
from tqdm import tqdm


CRITERIA = [
    "Clarity", "Specificity", "Completeness",
    "Safety", "Answerability", "Conciseness",
    "FormatConsistency",
]

EVAL_LINE_RE = re.compile(
    r"^\*\s*(?P<criterion>Clarity|Specificity|Completeness|Safety|Answerability|Conciseness|FormatConsistency)\s*:\s*(?P<score>[1-5])\/5\s*-\s*(?P<note>.+)$"
)


def is_valid_evaluation_block(text: str) -> bool:
    if not text:
        return False
    lines = [ln.strip() for ln in text.strip().splitlines()]
    try:
        start_idx = lines.index("Evaluation:")
    except ValueError:
        return False

    body = [ln for ln in lines[start_idx + 1:] if ln]
    matched = []
    for ln in body:
        m = EVAL_LINE_RE.match(ln)
        if m:
            matched.append(m.group("criterion"))

    return sorted(set(matched)) == sorted(CRITERIA) and len(matched) == 7


def parse_evaluation_to_json(evaluation_text: str) -> Optional[dict]:
    if not is_valid_evaluation_block(evaluation_text):
        return None

    parsed_data = {}
    pattern = re.compile(r"\*\s*(?P<criterion>\w+):\s*(?P<score>\d)/5\s*-\s*(?P<note>.+)")

    lines = evaluation_text.strip().split("\n")
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
            criterion = data["criterion"]
            score = int(data["score"])
            note = data["note"].strip()
            parsed_data[criterion] = {"score": score, "note": note}

    if sorted(parsed_data.keys()) != sorted(CRITERIA):
        return None

    return parsed_data


def build_evaluation_prompt(original_instruction: str) -> str:
    return f"""You are an evaluator. Given an Original Instruction,
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
{original_instruction}
"""


def generate_response_openai(
    client: OpenAI,
    model: str,
    max_output_tokens: int,
    prompt: str,
) -> str:
    resp = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": "",
            },
            {"role": "user", "content": prompt},
        ],
        max_output_tokens=max_output_tokens,
    )
    return resp.output_text


def process_one_record(
    client: OpenAI,
    model: str,
    max_output_tokens: int,
    max_retries: int,
    record: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if "instruction" not in record or not isinstance(record["instruction"], str):
        return None

    new_record = copy.deepcopy(record)
    instruction = record["instruction"]

    original_json = None
    feedback_text = None

    for _ in range(max_retries):
        prompt = build_evaluation_prompt(instruction)
        feedback_text = generate_response_openai(
            client=client,
            model=model,
            max_output_tokens=max_output_tokens,
            prompt=prompt,
        )
        original_json = parse_evaluation_to_json(feedback_text)
        if original_json is not None:
            break

    if original_json is None:
        return None

    new_record["feedback"] = feedback_text
    new_record["feedback_json"] = original_json
    new_record["feedback_avg_score"] = (
        sum(v["score"] for v in original_json.values()) / len(original_json)
    )

    return new_record


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def validate_cfg(cfg: DictConfig) -> None:
    if not getattr(cfg, "input_path", None):
        raise ValueError("cfg.input_path is required")
    if not getattr(cfg, "output_path", None):
        raise ValueError("cfg.output_path is required")


@hydra.main(version_base=None, config_path="")
def main(cfg: DictConfig):
    print(f"Loaded config name: {OmegaConf.to_yaml(cfg).strip()}")

    validate_cfg(cfg)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("No OPENAI_API_KEY found in environment variables.")

    client = OpenAI(api_key=api_key)

    dataset = read_jsonl(cfg.input_path)
    n = len(dataset)

    results_by_idx: List[Optional[Dict[str, Any]]] = [None] * n

    def _worker(idx_record: Tuple[int, Dict[str, Any]]):
        idx, record = idx_record
        try:
            out = process_one_record(
                client=client,
                model=str(cfg.model),
                max_output_tokens=int(cfg.max_output_tokens),
                max_retries=int(cfg.max_retries),
                record=record,
            )
            if out is None and bool(cfg.keep_failed):
                failed = copy.deepcopy(record)
                failed["failure_reason"] = "invalid_instruction_or_feedback_parse_failed"
                return idx, failed
            return idx, out
        except Exception as e:
            if bool(cfg.keep_failed):
                failed = copy.deepcopy(record)
                failed["failure_reason"] = f"exception: {type(e).__name__}: {e}"
                return idx, failed
            return idx, None

    with ThreadPoolExecutor(max_workers=int(cfg.workers)) as ex:
        futures = [ex.submit(_worker, (i, r)) for i, r in enumerate(dataset)]
        for fut in tqdm(as_completed(futures), total=n, desc="Processing JSONL", dynamic_ncols=True):
            idx, out = fut.result()
            results_by_idx[idx] = out

    results = [r for r in results_by_idx if r is not None]
    write_jsonl(cfg.output_path, results)

    print(f"Done. wrote {len(results)}/{n} lines to: {cfg.output_path}")


if __name__ == "__main__":
    main()
