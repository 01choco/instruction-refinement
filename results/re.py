import json

# 파일 경로 설정
input_file = "/opt/dlami/nvme/home/hailolab_seohyeong/instruction-ref/results/inference-ultrafeedback50.jsonl"
error_file = "/opt/dlami/nvme/home/hailolab_seohyeong/instruction-ref/results/with_error.jsonl"
no_error_file = "/opt/dlami/nvme/home/hailolab_seohyeong/instruction-ref/results/without_error.jsonl"

# 결과 저장용 리스트
with_error = []
without_error = []

# JSONL 파일 읽기
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        responses = item.get("responses", [])
        
        # [ERROR] 문자열 확인
        if any("[ERROR]" in response for response in responses):
            with_error.append(item)
        else:
            without_error.append(item)

# [ERROR]가 있는 항목 저장
with open(error_file, 'w', encoding='utf-8') as f:
    for item in with_error:
        instruction = {"instruction": item.get("instruction", "")}
        f.write(json.dumps(instruction, ensure_ascii=False) + '\n')

# [ERROR]가 없는 항목 저장
with open(no_error_file, 'w', encoding='utf-8') as f:
    for item in without_error:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print("처리가 완료되었습니다.")