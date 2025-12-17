# Query Refinement 

## Environment Setting 
```
conda env create -f environment_armo.yml
conda env create -f environment_gen.yml
```

## How to start
### 1. API Call Experiment 
```
bash src/run_loops.sh
```

### 2. Module Experiment 
```
# Setup 
git clone https://huggingface.co/01choco/feedback
git clone https://huggingface.co/01choco/refiner
mkdir models
mv ./feedback ./models/
mv ./refiner ./models/

# Run Refine
bash src/run_loops_module.sh
```