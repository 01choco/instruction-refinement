eval "$(conda shell.bash hook)"
export PYTHONUNBUFFERED=1

CFG_PATH=../config
CFG_NAME=config-module

conda activate qref_gen
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/inference.py --config-path $CFG_PATH --config-name $CFG_NAME
conda deactivate 

# inference path 
INPUT_PATH=./results/inference-module-ultrafeedback50.jsonl
LOOP_DONE=0

for i in 0 1 2; do
    export LOOP_CNT=$i
    export INPUT_PATH=$INPUT_PATH

    conda activate qref_armo
    CUDA_VISIBLE_DEVICES=0 python -u src/feedback_armo.py --config-path $CFG_PATH --config-name $CFG_NAME

    CNT_FILE=$(python - <<'PY'
import os
cfg_feedback = os.environ.get("FEEDBACK_PATH_HINT","")
PY
)
    CNT=$(cat ./results/ultra50-module/min/_tmp_cnt_${i}.txt)
    conda deactivate
    conda activate qref_gen
    # CUDA_VISIBLE_DEVICES=0,1 python -u src/feedback_module.py --config-path $CFG_PATH --config-name $CFG_NAME
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -u src/_feedback_gen.py --config-path $CFG_PATH --config-name $CFG_NAME
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -u src/_refine_gen.py --config-path $CFG_PATH --config-name $CFG_NAME
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -u src/_response_gen.py --config-path $CFG_PATH --config-name $CFG_NAME

    INPUT_PATH=./results/ultra50-module/min/_tmp_next_${i}.jsonl
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
