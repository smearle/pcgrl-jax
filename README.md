# PCGRL-jax

## Install

```
pip install -r requirements.txt
```

Then [install jax](https://jax.readthedocs.io/en/latest/installation.html):

## Training

To train a model, run:
```
python train.py
```
Arguments (pass these by running, e.g., `python train.py overwrite=True`):
- `overwrite`, bool, default=False`
    Whether to overwrite the model if it already exists.
- `render_freq`, int, default=100
    How often to render the environment.

During training, we render a few episodes to see how the model is doing (every `render_freq` updates). We use the same 
random seeds when resetting the environment, so that initial level layouts are the same between rounds of rendering.

## Hyperparameter sweeps

To train a sweep of models, run:
```
python sweep.py
```

This will perform grid searches over the groups of hyperparameters defined in `hypers`.

Arguments:
- `mode`, string, default=`train`, what type of jobs to launch while sweeping. Options are:
    - `train` trains the model for each experiment. Will attempt to re-load existing checkpoints by default.
    - `eval` evaluates each model in the sweep, given the same environment parameters as were seen during training.
    - `eval_cp` evaluates each model over a range of permitted change percentages.
    - `plot` iterates  plot the results of the sweep.
- `slurm`, bool, default=True
    Whether to submit each job in the sweep to a SLURM cluster (using the [submitit](https://github.com/facebookincubator/submitit) package)