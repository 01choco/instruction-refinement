# Instruction Refinement

This repository provides a codebase for running and reproducing Instruction Refinement experiments.  

---

## Environment Setup

Different Conda environments are used depending on loop stage. 

    conda env create -f environment.yml
    conda env create -f environment_armo.yml
    conda env create -f environment_gen.yml

- environment.yml (pref) : Environment for sampling and finalize your datasets
- environment_armo.yml (pref_gen): Environment for API call–based Feedback generation
- environment_gen.yml (pref_armo) : Environment for vllm model response generation 

---

## How to Run

### 1. Sampling Instructions 
Configurate your dataset, and sampling instructions in size you want.
```
conda activate pref
python src/sampling.py --config-name config --config-path ../config
python src/sampling-instruction --config-name config --config-path ../config
```

### 2. Run the Instruction Refinement Loop
Configurate your refinement loop in `/config/` and run your loops.
Example config file is provided in: `/config/config.yaml`

```
bash src/run_loops.sh
```

### 3. Evaluate 
Evaluate finalized refined / original dataset.
Code is provided in `/eval`. 
Configuration files are provided in : `\config\xeval.yaml` and `\config\yeval.yaml`.
```
conda activate pref
python src/xeval.py --config-name config-xeval --config-path ../config
python src/yeval.py --config-name config-yeval --config-path ../config
```

---

## Notes

- Each experiment script can be executed independently.
- Refer to scripts under `config/` for detailed parameter configurations.

---

## License

Research use only  
