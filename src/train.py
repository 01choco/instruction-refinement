import yaml
import copy
import subprocess
import hydra
import os
import sys

def create_yaml(cfg, template, data_name, loop_cnt):
    beta, lr, ratio = cfg.beta, cfg.lr, cfg.ratio
    if cfg.type == "dpo":
        print(f"Creating {cfg.model}_{cfg.type}_{data_name}_{loop_cnt}.yaml with beta={beta}, lr={lr}")

    elif cfg.type == "simpo":
        print(f"Creating {cfg.model}_{cfg.type}_{data_name}_{loop_cnt}.yaml with beta={beta}, lr={lr}, ratio={ratio}")

    new = copy.deepcopy(template)
    if loop_cnt == 0:
        new['model_name_or_path'] = 'princeton-nlp/Llama-3-Base-8B-SFT'
    else:
        new['model_name_or_path'] = f'../../../../../data1/dataset_cartography/{cfg.model_path}/{cfg.model}_{cfg.type}_{data_name}_{loop_cnt-1}'
    new['pref_beta'] = float(beta)
    new['learning_rate'] = float(lr)
    new['dataset'] = data_name
    new['output_dir'] = f'../../../../../data1/dataset_cartography/{cfg.save_path}/{cfg.model}-{cfg.type}-{data_name}-{loop_cnt}'
    new['save_steps'] = cfg.save_steps
    new['num_train_epochs'] = float(cfg.epoch)
    if cfg.type == "simpo":
        new['simpo_gamma'] = float(ratio)

    filename = f'./LLaMA-Factory/{cfg.yaml_path}/{cfg.model}_{cfg.type}_{data_name}_{loop_cnt}.yaml'

    # save .yaml files
    with open(filename, 'w') as f:
        yaml.dump(new, f, sort_keys=False)

    print(f"Created .yaml file: {filename}")

def train_model(cfg, data_name, loop_cnt):
    print(f"Training model using {cfg.model}_{cfg.type}_{data_name}_{loop_cnt}.yaml")

    cwd = os.getcwd()
    os.chdir("LLaMA-Factory")

    command = f"CUDA_VISIBLE_DEVICES={cfg.avail_devices} PYTHONPATH=./src llamafactory-cli train {cfg.yaml_path}/{cfg.model}_{cfg.type}_{data_name}_{loop_cnt}.yaml"
    subprocess.run(command, shell=True)

    os.chdir(cwd)

def export_model(cfg, template, data_name, loop_cnt):
    beta, lr, ratio = cfg.beta, cfg.lr, cfg.ratio
    print(f"Exporting model using {cfg.model}_{cfg.type}_{data_name}_{loop_cnt}.yaml")

    cwd = os.getcwd()
    os.chdir("LLaMA-Factory")

    adapter_name = f'{cfg.model}-{cfg.type}-{data_name}-{loop_cnt}'
    adapter_path = f'{cfg.save_path}/{adapter_name}'
    export_dir = f'../../../../../data1/dataset_cartography/{cfg.model_path}/{cfg.model}_{cfg.type}_{data_name}_{loop_cnt}'

    print(f"Creating {cfg.model}_{cfg.type}_{data_name}_{loop_cnt}.yaml")
    new = copy.deepcopy(template)
    if loop_cnt == 0:
        new['model_name_or_path'] = 'princeton-nlp/Llama-3-Base-8B-SFT'
    else:
        new['model_name_or_path'] = f'../../../../../data1/dataset_cartography/{cfg.model_path}/{cfg.model}_{cfg.type}_{data_name}_{loop_cnt-1}'
    new['adapter_name_or_path'] = f'../../../../../data1/dataset_cartography/{adapter_path}'
    new['export_dir'] = f'{export_dir}'

    filename = f'{cfg.export_yaml_path}/{cfg.model}_{cfg.type}_{data_name}_{loop_cnt}.yaml'

    # yaml file save
    with open(filename, 'w') as f:
        yaml.dump(new, f, sort_keys=False)

    print(f"Exporting {filename}")
    export_command = f"CUDA_VISIBLE_DEVICES={cfg.avail_devices} PYTHONPATH=./src llamafactory-cli export {filename}"
    subprocess.run(export_command, shell=True)

    os.chdir(cwd)

    return export_dir
