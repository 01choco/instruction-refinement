# Query Refinement

This repository provides a research-oriented codebase for running and reproducing Query Refinement experiments  
It supports two execution modes: API-based experiments and module-based experiments

---

## Environment Setup

Different Conda environments are used depending on the experiment type

    conda env create -f environment_armo.yml
    conda env create -f environment_gen.yml

- environment_armo.yml: Environment for API call–based experiments
- environment_gen.yml: Environment for local module–based experiments

---

## How to Run

### 1. API Call Experiment

Run the query refinement loop using direct API calls

    bash src/run_loops.sh

- Uses external APIs
- Suitable for lightweight and fast iteration

---

### 2. Module Experiment

Run experiments using locally loaded refinement modules

#### Setup

Clone the required models from Hugging Face

    git clone https://huggingface.co/01choco/feedback
    git clone https://huggingface.co/01choco/refiner

    mkdir models
    mv ./feedback ./models/
    mv ./refiner ./models/

Expected directory structure

    models/
     ├─ feedback/
     └─ refiner/

#### Run

    bash src/run_loops_module.sh

- Uses local model modules
- Suitable for repeated experiments and debugging

---

## Notes

- Each experiment script can be executed independently
- Refer to scripts under src/ for detailed parameter configurations
- Models and weights are managed via Hugging Face repositories

---

## License

Research use only  
For commercial use, please contact the author
