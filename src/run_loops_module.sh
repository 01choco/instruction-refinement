eval "$(conda shell.bash hook)"
export PYTHONUNBUFFERED=1

CFG_PATH=../config
CFG_NAME=config-module

# inference path 
INPUT_PATH=./results/inference-ultrafeedback50.jsonl
LOOP_DONE=0

for i in 0 1 2; do
    export LOOP_CNT=$i
    export INPUT_PATH=$INPUT_PATH

    conda activate qref_armo
    CUDA_VISIBLE_DEVICES=2 python -u src/feedback_armo.py --config-path $CFG_PATH --config-name $CFG_NAME

    CNT_FILE=$(python - <<'PY'
import os
cfg_feedback = os.environ.get("FEEDBACK_PATH_HINT","")
PY
)
    CNT=$(cat ./results/feedback-50-error/min/_tmp_cnt_${i}.txt)
    conda deavtivate
    conda activate qref_gen
    CUDA_VISIBLE_DEVICES=2 python -u src/feedback_module.py --config-path $CFG_PATH --config-name $CFG_NAME

    INPUT_PATH=./results/feedback-50-error/min/_tmp_next_${i}.jsonl
    LOOP_DONE=$((i+1))

    if [ "$CNT" = "0" ]; then
        break
    fi
done

# finalize
export LOOP_DONE=$LOOP_DONE
export FINAL_DATASET_PATH=$INPUT_PATH
conda activate qref_gen
python -u src/file_generation.py --config-path $CFG_PATH --config-name $CFG_NAME
