#!/usr/bin/env python3
import argparse
import os
import sys
import glob
import hydra
import traceback
from shutil import copyfile
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

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
    exclude_dir  = ['data', '__pycache__', 'log', '.git', 'res', 'check']
    exclude_file = ['cfg', 'cmd', '.gitignore']
    exclude_ext  = ['.png', '.jpg', '.pt', '.npz']
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

    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = f"{args.proc.port}"
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" if args.proc.cpu \
    else ','.join([str(gpu_num) for gpu_num in args.proc.gpus])
    ''' The CUDA_VISIBLE_DEVICES environment variable is read by the cuda driver.
        So it needs to be set before the cuda driver is initialized.
        It is best if you make sure it is set **before** importing torch
        (or at least before you do anything cuda related in torch).
        source: https://discuss.pytorch.org/t/os-environ-cuda-visible-devices-not-functioning/105545/4
    '''
    import torch
    if not args.proc.train:
        # This is redundant in the case of being `args.proc.train == True`,
        # since Lightning will seed everything (see `src/trainer.py`.)
        torch.manual_seed(args.proc.seed)

    args.cwd = HydraConfig.get().runtime.cwd

    if args.task.save_name is not None:
        save_dir_name = args.task.save_name
    elif args.proc.debug or args.task.result_dir=='debug':
        args.proc.debug = True
        save_dir_name = 'debug'
    else:
        save_dir_name = args.task.result_dir
    save_dir = f'{args.task.root_dir}/{save_dir_name}'

    if args.task.measure_time:
        args.task.plot = False
        args.task.save = False
        args.task.plot_state = False

    if args.task.result_dir == "debug":
        args.proc.debug = True

    if args.proc.simulate or args.proc.train:
        backup_code(args)

    if args.proc.simulate:
        model_name = 'random' if args.model.excitation is None else args.model.excitation
        n_samples = args.task.num_samples // args.task.batch_size
        from src.task import simulate
        # run simulation
        simulate.run(args, save_dir, model_name, n_samples=n_samples)

    if args.proc.evaluate:
        from src.task import evaluate
        # evaluate simulation results
        load_dir = save_dir if args.task.load_dir is None else args.task.load_dir
        evaluate.evaluate(load_dir)

    if args.proc.summarize:
        from src.task import summarize
        # summarize evaluation results
        load_dir = save_dir if args.task.load_dir is None else args.task.load_dir
        summarize.summarize(load_dir)

    if args.proc.process_training_data:
        from src.task import process_training_data
        # preprocess simulation results as training data
        process_training_data.process(args)

    if args.proc.train:
        from src import trainer
        # train neural network
        trainer.train(args)

if __name__=='__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    main()

