#!/usr/bin/env python3
import argparse
import os
import glob
import hydra
from datetime import datetime
from shutil import copyfile
import sys
import traceback
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from src.task import simulate, evaluate, summarize
from src.utils import config as cf

class ConfigArgument:
    def __getitem__(self,key):
        return getattr(self, key)
    def __setitem__(self,key,value):
        return setattr(self, key, value)

def get_object(config, m):
    for key in config.keys():
        if isinstance(config[key], DictConfig):
            m[key] = ConfigArgument()
            get_object(config[key], m[key])
        else:
            m[key] = config[key]
    return m

def backup_code(args):
    # Copy directory sturcture and files
    exclude_dir  = ['data', '__pycache__', 'log', '.git']
    exclude_file = ['cfg', 'cmd', '.gitignore']
    exclude_ext  = ['.png', '.jpg', '.pt']
    filepath = []
    for dirpath, dirnames, filenames in os.walk(args.cwd, topdown=True):
        if not any(dir in dirpath for dir in exclude_dir):
            filtered_files=[name for name in filenames if (os.path.splitext(name)[-1] not in exclude_ext) and (name not in exclude_file)]
            filepath.append({'dir': dirpath, 'files': filtered_files})

    num_strip = len(args.cwd)
    for path in filepath:
        dirname = path['dir'][num_strip+1:]
        for filename in path['files']:
            if '.swp' in filename or '.onnx' in filename:
                continue
            file2copy = os.path.join(path['dir'], filename)
            os.makedirs(f"codes/{dirname}", exist_ok=True)
            filepath2save = os.path.join(f"codes/{dirname}", filename)
            copyfile(file2copy, filepath2save)

@hydra.main(config_path="src/configs", config_name="config.yaml")
def main(config: OmegaConf):
    config = cf.process_config(config)
    cf.print_config(config, resolve=True)
    args = get_object(config, ConfigArgument())

    os.environ["CUDA_VISIBLE_DEVICES"] = '-1' if args.proc.cpu else ','.join([str(gpu_num) for gpu_num in args.proc.gpus])
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = f"{args.proc.port}"
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    args.cwd = HydraConfig.get().runtime.cwd

    model_name = 'random' if args.model.excitation is None else args.model.excitation
    save_dir_name = args.task.save_name if args.task.save_name is not None else args.task.result_dir if not args.proc.debug else 'debug'
    save_dir = f'{args.task.root_dir}/{save_dir_name}'
    n_samples = args.task.num_samples // args.task.batch_size

    if args.task.measure_time:
        args.task.plot = False
        args.task.save = False
        args.task.plot_state = False

    if args.task.result_dir == "debug":
        args.proc.debug = True

    backup_code(args)
    if args.proc.run:
        # run simulation
        simulate.run(args, save_dir, model_name, n_samples=n_samples)

    if args.proc.evaluate:
        # evaluate simulation results
        load_dir = save_dir if args.task.load_dir is None else args.task.load_dir
        evaluate.evaluate(load_dir)

    if args.proc.summarize:
        # summarize evaluation results
        load_dir = save_dir if args.task.load_dir is None else args.task.load_dir
        summarize.summarize(load_dir)

if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    main()

