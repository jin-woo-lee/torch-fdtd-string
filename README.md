<p align="center">
<img src="res/2024-string.gif" width="24em">
<br>
Simulated string with bowing excitation, exhibiting a Helmholtz motion.
</p>

----

This repo contains two PyTorch-based string simulators,
*StringFDTD-Torch* [[1]](#1) and *DMSP* [[2]](#2).

- StringFDTD-Torch [![arXiv](https://img.shields.io/badge/arXiv-2311.18505-b31b1b.svg)](https://arxiv.org/abs/2311.18505)
   + *StringFDTD-Torch* is a planar string simulation engine
     for musical instrument sound synthesis research.
     It simulates a string vibration from a given set
     of mechanical properties and excitation conditions
     based on a finite difference scheme
     (i.e., finite difference time domain)
     and outputs the resulting string sound.

- DMSP [![OpenReview](https://img.shields.io/badge/OpenReview-fpxRpPbF1t-b31b1b.svg)](https://openreview.net/forum?id=fpxRpPbF1t) [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/szin94/dmsp)
   + *Differentiable Modal Synthesis for Physical Modeling (DMSP)*
     is a neural network trained to approximate the string motion
     simulated using StringFDTD-Torch but in an efficient manner
     augmenting the modal synthesis method in a similar manner to
     the DDSP approach.  

## 1. StringFDTD-Torch
StringFDTD-Torch can simulate under your own
predefined configurations or can also be used
as a dataset generator by randomly augmenting
within a set of bounds for the mechanical conditions.
It can also simulate on the CPU or GPU,
depending on the state of the device you want to simulate.
You may want to pick CPU if you are
simulating your own configuration in a small batch size.
However, it is recommended to use GPU for simulating
as a dataset generator with large batch size,
to benefit from the GPU parallelism.

### 1.1. Dependencies
Install python and system dependencies 
```bash
xargs apt-get install -y < apt-packages.txt
pip install -r requirements.txt
```

### 1.2. Running as a dataset generator
StringFDTD-Torch can simulate the string under various conditions.
The configuration can be specified by passing arguments or
by defining them in a `.yaml` file (or using both of them.)

For example, we have prepared a typical example configuration as
`src/configs/experiment/nsynth-like.yaml`.
The `nsynth-like.yaml` predefines the configurations for
generating sound samples akin to NSynth [[4]](#4).
You can run the simulation with the following command.
```bash
python -m run \
    experiment=nsynth-like \
    task.result_dir=my_fdtd_simulation
```
This will first load the base config file at `src/configs/config.yaml`,
then override it with the preset config file at `src/configs/experiment/nsynth-like.json`,
dump its `task.result_dir` argument to `my_fdtd_simulation` as specified by the user,
and finally run the `run.py` file.

Everything the `StringFDTD-Torch` simulates will be saved under `{{ task.root_dir }}/{{ task.result_dir }}` directory.
By default, the `task.root_dir` is set to `./results` and if you run the above command,
the results will be saved under `./results/my_fdtd_simulation` directory.
The saved files will be organized as follows:
```bash
my_fdtd_simulation/
├── codes/                     # source code backup for the current simulation
├── config_tree.txt            # final configuration tree (saves all the passed arguments as a final config)
├── cpu_time.txt               # measured time for simulation
├── run.log                    # log file
├── {batch_id}-{batch_number}/ # simulation results (batch_id is a random 8-character string)
│   ├── bow_params.npz         # properties related to bowing excitation
│   ├── hammer_params.npz      # properties related to hammering excitation
│   ├── output-u.wav           # pickup signal for the lateral motion
│   ├── output-z.wav           # pickup signal for the longitudinal motion
│   ├── output.wav             # mixed signal of output-u and output-z
│   ├── simulation.npz         # simulated string states for whole space
│   ├── simulation_config.yaml # simulation parameters summary
│   └── string_params.npz      # properties related to strings
└── ...
```

### 1.3. Running with a specified conditions
In order to provide specific conditions
(such as predefined F0, bowing force, hammering timing, etc.)
save them as `/path/to/preset/*.npy` file.
Passing `task.load_config=/path/to/preset` will load
the preset parameters for simulation.
The other parameters, those are not defined in
`/path/to/preset`, will be randomly augmented.
You can also specify the augmentation range
using the `experiment` argument as follows.
```bash
python -m run experiment=nsynth-like task.excitation=hammer task.load_config=data/trumpet
```

### 1.4. Notes on the FDTD simulation
The simulation is based on the finite difference time domain (FDTD) method.
The FDTD method is a numerical approach to solve the wave equation
for the string motion. The wave equation is discretized in both time and space,
and the string motion is computed iteratively over time steps.
Here are some important notes regarding the FDTD simulation:

#### 1.4.1. Devices
StringFDTD-Torch uses GPUs by default.
In order to simulate using CPUs, set `proc.cpu=true`.
You might need to `export CUDA_VISIBLE_DEVICES=""` in your bash script.
A typical usage example, along with specifying the processors via taskset,
would appear like this.

```bash
taskset -c 16-31 python -m run proc.cpu=true ...
```

#### 1.4.2. Fundamental frequency pre-correction
In the string instrument simulation, the pitch (or the fundamental frequency) of the sound can vary depending on the stiffness of the string.  While pitch changes with stiffness have been and continue to be studied for a long time, we have found that the amount of detune in the simulated sounds closely matches the theoretical model proposed by Fletcher [[3]](#3). And even if detune is a natural phenomenon, we provide the ability to pre-correct the pitch based on the theoretical model to achieve the sound at the intended pitch.

<p align="center">
<img src="res/precorrect.png" width="500">
</p>

As the spectrograms above show, without the precorrection, there is a bit of detune in the intended fundamental frequency (white dashed lines) and the estimated one. However, the detune is resolved by pre-correcting the input fundamental frequency. Utilizing the pre-correction feature significantly reduces the detune phenomenon, but at the cost of a relatively large increase in computation and speed. So, if you think this pre-correction is not necessary, you can turn off pre-correction by passing `task.precorrect=false` argument. Please note that this argument is set `true` by default.

```bash
python -m run  ... task.precorrect=false
```

#### 1.4.3. Debug mode
You can run the simulation in a debug mode by passing `task.result_dir=debug`.
```bash
python -m run experiment=my_experiment task.result_dir=debug
```

#### 1.4.4. Evaluate or plot the simulated results
You can evaluate or plot the simulation results by passing `experiment=evaluate`.
Please specify the path to the directory that contains the simulation results.
```bash
ls /path/to/simulated/directory # 00000000-0  00000000-1  ...  run.log
python -m run experiment=evaluate task.load_dir='./my_fdtd_simulation' 
```


## 2. DMSP
The DMSP model [[2]](#2) can be trained using the data simulated as above.

### 2.1. Preprocess the FDTD data
In order to train the DMSP model, first preprocess the simulated FDTD data
following the `configs/experiment/process_training_data.yaml`.
This incorporates spatially upsampling the FDTD results and saving the modal solutions.
<details>
  <summary>Processing the whole FDTD data</summary>

  ```bash
  python -m run experiment=process_training_data \
      task.result_dir=my_fdtd_simulation \
      task.save_dir=my_dmsp_data
  ```
  The processed data will be saved under `{{ task.root_dir }}/my_dmsp_data`.
  The saved files will be organized as follows:
  ```bash
  my_dmsp_data/
  ├── {batch_id}-{batch_number}/ # same as the simulation results
  │   ├── parameters.npz         # parameters for the simulation
  │   ├── ua-{000}.wav           # 1D modal solution of the string at x={000}
  │   ├── ...
  │   ├── ut-{000}.wav           # Lateral FDTD solution of the string at x={000}
  │   ├── ...
  │   └── vt.wav                 # Velocity of the lateral solution summed up over x.
  └── ...
  ```
</details>
<details>
  <summary>Processing with train/validation/test split</summary>

  The `process_training_data.yaml` config file also supports splitting the data into
  training, validation, and test sets. You can specify the split ratio using the `task.data_split` and the `task.split_n` arguments.
  - `data_split`: splits the list of whole data into `data_split` parts (set 0 to disable)
  - `split_n`: only processes the `split_n`-th part of the splitted data sublist
  For example, if you want to split the data into 5 parts and process the first part as test data,
  the second part as validation data, and the remaining three parts as training data, you can run the following commands:
  ```bash
  nohup python -m run proc.gpus=[0] experiment=process_training_data task.result_dir=my_fdtd_simulation \
      task.save_dir=my_dmsp_data/test  task.data_split=5 task.split_n=0 > log_test &
  nohup python -m run proc.gpus=[0] experiment=process_training_data task.result_dir=my_fdtd_simulation \
      task.save_dir=my_dmsp_data/valid task.data_split=5 task.split_n=1 > log_valid &
  nohup python -m run proc.gpus=[0] experiment=process_training_data task.result_dir=my_fdtd_simulation \
      task.save_dir=my_dmsp_data/train task.data_split=5 task.split_n=2 > log_train1 &
  nohup python -m run proc.gpus=[0] experiment=process_training_data task.result_dir=my_fdtd_simulation \
      task.save_dir=my_dmsp_data/train task.data_split=5 task.split_n=3 > log_train2 &
  nohup python -m run proc.gpus=[0] experiment=process_training_data task.result_dir=my_fdtd_simulation \
      task.save_dir=my_dmsp_data/train task.data_split=5 task.split_n=4 > log_train3 &
  ```
  The processed data will be saved under `{{ task.root_dir }}/my_dmsp_data/{{ split }}`
  where `{{ split }}` is the name of the split (e.g., `train`, `valid`, `test`).
  You can run the above command in parallel to speed up the processing, but make sure to
  secure enough memory for each process. The saved files will be organized as follows:
  ```bash
  my_dmsp_data/{{ split }}/
  ├── {batch_id}-{batch_number}/ # same as the simulation results
  │   ├── parameters.npz         # parameters for the simulation
  │   ├── ua-{000}.wav           # 1D modal solution of the string at x={000}
  │   ├── ...
  │   ├── ut-{000}.wav           # Lateral FDTD solution of the string at x={000}
  │   ├── ...
  │   └── vt.wav                 # Velocity of the lateral solution summed up over x.
  └── ...
  ```
</details>
See `src/configs/experiment/process_training_data.yaml` for more details.


### 2.2. Train the DMSP model
Now train the DMSP model using the preset saved under `configs/experiment/synth-dmsp.yaml`.
```bash
python -m run proc.gpus=[0] experiment=synth-dmsp \
    task.load_dir=./results \
    task.load_name=my_dmsp_data
```
For more details, see `src/configs/experiment/synth-dmsp.yaml`.
The train/valid/test results can be found under `{{ task.root_dir }}/{{ task.result_dir }}`.
The saved files will be organized as follows:
```bash
{{ task.root_dir }}/{{ task.result_dir }}/
├── codes/           # source code backup for the current training
├── config_tree.txt  # the final configuration tree
├── run.log          # running log
├── string/          # checkpoints
│   └── {run_id}/checkpoints/'epoch={epoch}-step={step}.ckpt'
├── test/            # test results
└── valid/           # validation results
    ├── plot/
    │   ├── {epoch}-{iteration}-{batch_number}.png
    │   └── ...
    └── wave/
        ├── {epoch}-{iteration}-{batch_number}.wav
        └── ...
```

### 2.3. Inference
To run inference using the trained DMSP model,
you can use the same `synth-dmsp` config file as a base.
```bash
python -m run proc.gpus=[0] experiment=synth-dmsp \
    task.load_dir=./results task.load_name=my_dmsp_data \
    hydra.run.dir=/full/path/to/your/trained/{{ task.result_dir }} \
    proc.train=false proc.test=true \
    task.plot_test_video=true \
    task.save_test_score=true
```
Provide the `hydra.run.dir` argument to specify the full path to the directory
where the trained model is saved.

When you run the above command, here are what happens:
- Configuration files are loaded in the following order:
  - Load the base configuration file at `src/configs/config.yaml`
  - Override it with the preset config file at `src/configs/experiment/synth-dmsp.yaml`
  - Override it with the in-line arguments (e.g., `task.load_dir`, `task.load_name`, `hydra.run.dir`, etc.)
- The network is defined and loaded in the following order: 
  - Source code is loaded from the `{{ hydra.run.dir }}/codes` directory
  - Model checkpoint is loaded from `{{ hydra.run.dir }}/string/{run_id}/checkpoints/*.ckpt`.
- The test step is run as follows:
  - The test data is loaded from `{{ task.load_dir }}/{{ task.load_name }}/` directory.
  - The inferenced outputs are saved under `{{ hydra.run.dir }}/test/` directory.
Note that the inference always use the (backup) code saved under `{{ hydra.run.dir }}/codes` directory.
This ensures that the model is evaluated on the exact same version of the source code as the code it was trained on.

The saved files will be organized as follows:
```bash
{{ hydra.run.dir }}/
├── codes/           # source code backup for the current training
├── config_tree.txt  # the final configuration tree
├── run.log          # running log
├── string/          # checkpoints
├── test/            # test results
│   ├── {{ task.load_name }}/
│   │   ├── score/          # test scores (when provided `task.save_test_score=true`)
│   │   │   ├── modal.txt   # Modal synthesis's score
│   │   │   └── output.txt  # Current model's output score
│   │   ├── state/                  # test output plot (when provided `task.plot_test_video=true`)
│   │   │   ├── {epoch}-{batch}.npz # string displacement raw file
│   │   │   ├── {epoch}-{batch}.pdf # string displacement plot
│   │   │   └── ...
│   │   └── video/                    # test output video (when provided `task.plot_test_video=true`)
│   │       ├── {epoch}-{batch}-{method}-silent_video.mp4 # string motion video without sound
│   │       ├── {epoch}-{batch}-{method}.mp4              # string motion video with sound
│   │       ├── {epoch}-{batch}-{method}.pdf              # string motion sample plot
│   │       └── ...
│   ├── plot/
│   └── wave/
└── valid/           # validation results
```
Note that the results from the same model for different `{{ task.load_name }}`s
will be distinguished by the different `{{ task.load_name }}` directory names.


### 2.4. Notes on the DMSP synthesizer
The DMSP synthesizer is a neural network model
that approximates the string motion simulated using StringFDTD-Torch.
It is trained to learn the mapping between the input parameters
and the output string motion.
The DMSP model is based on the modal synthesis method,
which decomposes the string motion into a sum of modal vibrations.

#### 2.4.1. DMSP-Hybrid model
In acquiring the eigenvalues and eigenvectors of the string motion,
the model can either leverage the Newton method or a NN trained to approximate the Newton method.
We call the latter as `DMSP` and the former as `DMSP-Hybrid` in the paper [[2]](#2).
Both share the same architecture and training process,
but the model can be inferred using the Newton method by passing
`model.use_precomputed_mode=true` argument, making the result as the DMSP-Hybrid model's output.
```bash
python -m run proc.gpus=[0] experiment=synth-dmsp \
    task.load_dir=./results task.load_name=my_dmsp_data proc.train=false proc.test=true \
    model.use_precomputed_mode=true
```

#### 2.4.2. Debug mode
You can run the training in a debug mode by passing `task.result_dir=debug`.
```bash
python -m run experiment=my_experiment task.result_dir=debug
```

#### 2.4.3. Development environment
The code is developed and tested on the following environment:
```bash
python 3.9.21
gcc 12.2.0
```

## 3. Citation

If you use this code in your research, please cite the following paper.

```bib
@inproceedings{leedifferentiable,
  title     = {Differentiable Modal Synthesis for Physical Modeling of Planar String Sound and Motion Simulation},
  author    = {Lee, Jin Woo and Park, Jaehyun and Choi, Min Jun and Lee, Kyogu},
  booktitle = {The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS)},
  year      = {2024}
}
@inproceedings{lee2024string,
  title        = {String Sound Synthesize on GPU-accelerated Finite Difference Scheme},
  author       = {Lee, Jin Woo and Choi, Min Jun and Lee, Kyogu},
  booktitle    = {ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages        = {1--5},
  year         = {2024},
  organization = {IEEE}
}
```

## 4. References
<a id="1">[1]</a> 
Lee, J. W., Choi, M. J., & Lee, K. (2024, April).
String Sound Synthesizer On Gpu-Accelerated Finite Difference Scheme.
In *ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)* (pp. 1491-1495). IEEE.

<a id="2">[2]</a> 
Lee, J. W., Park, J., Choi, M. J., & Lee, K. (2024).
Differentiable Modal Synthesis for Physical Modeling of Planar String Sound and Motion Simulation.
In *The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS)*.

<a id="3">[3]</a> 
Fletcher, H. (1964).
Normal vibration frequencies of a stiff piano string.
*The Journal of the Acoustical Society of America*, 36(1), 203-209.

<a id="4">[4]</a> 
Engel, J., Resnick, C., Roberts, A., Dieleman, S., Norouzi, M., Eck, D., & Simonyan, K. (2017, July).
Neural audio synthesis of musical notes with wavenet autoencoders.
In *International Conference on Machine Learning* (pp. 1068-1077). PMLR.



