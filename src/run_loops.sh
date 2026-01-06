eval "$(conda shell.bash hook)"
CUDA_VISIBLE_DEVICES=0,1 # set your device 
CFG_PATH=../config
CFG_NAME=config # your config file name 

conda activate qref_gen
python src/inference.py --config-path $CFG_PATH --config-name $CFG_NAME
conda deactivate 

# inference path 
INPUT_PATH=./results/inference-path.jsonl
LOOP_DONE=0

# refinement loop starts 
for i in 0 1 2; do
    export LOOP_CNT=$i
    export INPUT_PATH=$INPUT_PATH

    conda activate qref_armo
    python -u src/feedback_armo.py --config-path $CFG_PATH --config-name $CFG_NAME
    conda deactivate

    CNT_FILE=$(python - <<'PY'
import os
cfg_feedback = os.environ.get("FEEDBACK_PATH_HINT","")
PY
)
    CNT=$(cat ./results/feedback-path/_tmp_cnt_${i}.txt)

    conda activate qref_gen
    python -u src/feedback_gen.py --config-path $CFG_PATH --config-name $CFG_NAME
    conda deactivate

    INPUT_PATH=./results/feedback-path/_tmp_next_${i}.jsonl
    LOOP_DONE=$((i+1))

    if [ "$CNT" = "0" ]; then
        break
    fi
done

# final file generation
export LOOP_DONE=$LOOP_DONE
export FINAL_DATASET_PATH=$INPUT_PATH

conda activate qref_gen
python -u src/file_generation.py --config-path $CFG_PATH --config-name $CFG_NAME
conda deactivate