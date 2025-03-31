import os
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import sys
import glob

import pytorch_lightning as pl
from lightning.pytorch import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from torchinfo import summary

def get_proj_dir(args):
    if not os.path.isabs(args.task.ckpt_dir):
        return f"{args.task.root_dir}/{args.task.ckpt_dir}"
    else:
        return args.task.ckpt_dir

def get_checkpoint(args):
    ''' args.task.ckpt_dir <-- args.task.result_dir '''
    proj_dir = get_proj_dir(args)
    ckpt_dir = f'{proj_dir}/{args.task.project}/*/checkpoints'
    best_ckpt_path = glob.glob(f'{ckpt_dir}/*.ckpt')
    assert len(best_ckpt_path) == 1, [best_ckpt_path, ckpt_dir]
    return os.path.join(best_ckpt_path[0])

def train(args):
    seed_everything(args.proc.seed, workers=True)
 
    scr = __import__(f'src.task.{args.task._name_}', fromlist=[''])
    model = scr.Trainer(args)
    if args.task.ckpt_dir is not None:
        ckpt_path = get_checkpoint(args)
        model = model.load_from_checkpoint(ckpt_path)
    summary(model)

    os.environ['WANDB_SILENT']="true"
    logger = WandbLogger(
        project=args.task.project,
        name=args.task.run,
        group=args.task._name_,
        dir=f".",
        mode = 'disabled' if args.proc.debug else 'online',
        allow_val_change=True,
        resume='allow',
        anonymous='allow',
        config=args,
    )

    from src import callbacks
    pl_callbacks = [ ]
    pl_callbacks += [ callbacks.PlotResults(args), ]
    pl_callbacks += [ LearningRateMonitor(logging_interval='step'), ] if not args.proc.debug else []

    mnum = min(args.proc.gpus)
    gpus=[gpu_num - mnum for gpu_num in args.proc.gpus]
    num_sanity_val_steps = 1 if args.proc.debug else 0
    #num_sanity_val_steps = 0

    pl_conf = dict(
        devices='auto',
        strategy='ddp_find_unused_parameters_true',
        num_sanity_val_steps=num_sanity_val_steps,
        logger=logger,
        callbacks=pl_callbacks,
        profiler="simple",
        max_epochs=args.task.total_epoch,
        check_val_every_n_epoch=args.task.valid_epoch,
        detect_anomaly=True if args.proc.debug else False,
    )
    #if args.task.overfit:
    #    pl_conf.update(dict(limit_train_batches=1))

    # train model
    trainer = pl.Trainer(**pl_conf)
    trainer.fit(model)

def eval(args):
    seed_everything(args.proc.seed, workers=True)
 
    print("*** Running in test mode!")

    proj_dir = get_proj_dir(args)
    sys.path.append(proj_dir)
    trainers_src = __import__(f'codes.src.task.{args.task._name_}', fromlist=[''])
    callback_src = __import__(f'codes.src.callbacks', fromlist=[''])

    ckpt_path = get_checkpoint(args)
    print(f"... load model ckpt from  : {ckpt_path}")
    #hpar_path = f"results.{args.task.result_dir}.lightning_logs"
    model = trainers_src.Trainer.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        #hparams_file=f"",
        map_location=None,
        args=args,
    )

    pl_callbacks = []
    if args.task.save_test_score:
        pl_callbacks += [ callback_src.SaveTestResults(args), ]
    if args.task.plot_test_video:
        pl_callbacks += [ callback_src.PlotStateVideo(args), ]
    mnum = min(args.proc.gpus)
    gpus=[gpu_num - mnum for gpu_num in args.proc.gpus]
    pl_conf = dict(
        devices='auto',
        strategy='ddp_find_unused_parameters_true',
        default_root_dir=proj_dir,
        logger=None,
        callbacks=pl_callbacks,
        profiler="simple",
        max_epochs=args.task.total_epoch,
        check_val_every_n_epoch=args.task.valid_epoch,
        detect_anomaly=True if args.proc.debug else False,
    )
    trainer = pl.Trainer(**pl_conf)
    trainer.test(model)


