# PCGRL-jax

## TODO:
<input type="checkbox" disabled checked />  add config hyperparam to control hidden nodes, allowing for comparison of *~equivalent-size* models with diff. receptive field sizes

<input type="checkbox" disabled checked /> controllable metric target bounds depend on actual map shape when randomizing map shape per-episode

<input type="checkbox" disabled checked/> eval on larger maps (32x32)
- <input type="checkbox" disabled/> eval on all possible map shapes within larger square map?

<input type="checkbox" disabled checked/> train on larger maps (32x32 if possible)

<input type="checkbox" disabled /> sparse reward (to compare against action shapes. Also might help with training on larger maps)

<input type="checkbox" disabled /> optimize pathfinding (jax.lax.conv)

<input type="checkbox" disabled /> new domains (treasure... more keys/doors?) and representations (turtle, re-implement O.G. wide model and compare 
 against NCA, FractalNet...)

<input type="checkbox" disabled /> make enemies chase agent (when agent is in "line of sight", move toward the player by 1 tile every 2 timesteps), 
  add combat mechanics


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

To save a `misc_stats.json` that records the number of timesteps for which a given mmodel has trained, we hackishly run `python sweep.py mode=plot slurm=False` (we're getting this info from the last row of the `progress.csv` used for plotting). Other stats are recorded when running with `mode=eval` or the like.