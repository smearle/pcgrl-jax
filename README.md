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
- 

## Hyper-parameter sweeps

To train a sweep of models, run:
```
python sweep.py
```
Arguments:
- `plot`, bool, default=False
    Whether to plot the results of the sweep.