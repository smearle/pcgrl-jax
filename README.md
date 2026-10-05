# PCGRL-jax

Procedural Content Generation via Reinforcement Learning (PCGRL), implemented end-to-end in JAX.

The level-generation environments, the PPO training loop and evaluation are all written in JAX, so thousands of
environments can run in parallel on one GPU. The codebase supports:

- **Single-agent PCGRL**: one generator agent edits a level, with `narrow`, `turtle`, `wide` and `nca` representations.
- **Multi-agent PCGRL**: several generator agents edit the same level at the same time (MAPPO-style training in `train_ma.py`).
- **Controllable generation**: you set target values for level metrics (e.g. path length, number of regions).
- **Generalization experiments**: you can vary the observation window, the map size and shape, the change percentage,
  the number of board scans and more, then cross-evaluate on out-of-distribution maps.

This repository contains the code for the two papers listed in [Citation](#citation).

## Installation

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`, Python 3.12):

```bash
uv sync --extra cuda12   # NVIDIA GPU (CUDA 12)
uv sync                  # CPU only
```

This creates `.venv/`. Run scripts either with `uv run python train.py ...` or after `source .venv/bin/activate`.
The commands below assume the environment is activated.

If you can't use uv, `uv export --extra cuda12 --no-hashes -o requirements.txt` gives a pinned file for `pip`.

Logging uses [Weights & Biases](https://wandb.ai). Pass `wandb_mode=disabled` (or `offline`) to turn it off.

## Quick start

All entry points are [Hydra](https://hydra.cc) apps, so you override any field of the dataclasses in
[`conf/config.py`](conf/config.py) from the command line:

```bash
# Single-agent: train a generator on the binary problem
python train.py problem=binary representation=narrow model=conv overwrite=True

# Multi-agent: 3 turtle agents sharing one map
python train_ma.py problem=binary n_agents=3 model=conv2 representation=turtle

# Render GIFs of a trained single-agent model (same args as used for training)
python enjoy.py problem=binary representation=narrow model=conv

# Evaluate a trained model
python eval.py problem=binary representation=narrow model=conv
python eval_ma.py problem=binary n_agents=3 model=conv2
```

Checkpoints, logs and renders go to `saves/<experiment-name>/`. The experiment name is derived from the config
(see `get_exp_dir` in [`utils.py`](utils.py)).

### Main options

| Option | Values | Notes |
|---|---|---|
| `problem` | `binary`, `maze`, `dungeon`, `dungeon2` | Level-design task; see [`envs/probs/`](envs/probs) |
| `representation` | `narrow`, `turtle`, `wide`, `nca` | How the agent edits the map; see [`envs/reps/`](envs/reps) |
| `model` | `conv`, `conv2`, `seqnca`, `rnn`, `dense`, `nca` | Policy architecture; see [`models.py`](models.py) |
| `map_width` | int (default `16`) | Size of the (square) map |
| `randomize_map_shape` | bool | Sample a random rectangular map shape every episode |
| `obs_size` / `arf_size` / `vrf_size` | int (`-1` = full map) | Observation window for the policy / action branch / value branch |
| `ctrl_metrics` | e.g. `[diameter,n_regions]` | Metrics whose targets the agent conditions on |
| `change_pct` | float (`-1` = off) | Maximum fraction of tiles the agent may change |
| `max_board_scans` | float | Episode length, measured in passes over the whole board |
| `n_agents` | int | Number of generator agents (multi-agent only) |
| `n_envs`, `total_timesteps`, `lr`, ... | | PPO hyperparameters |

## Experiment sweeps

Sweeps are grid searches defined as YAML files in [`conf/sweeps/`](conf/sweeps), e.g.
[`conf/sweeps/ma_board_scans_binary_aiide.yaml`](conf/sweeps/ma_board_scans_binary_aiide.yaml).
Each key lists the values to sweep over. `eval_hypers` lists extra settings to sweep over at evaluation time only.

```bash
# Train every config in a sweep (locally, one after another)
python sweep.py name=ma_board_scans_binary_aiide mode=train slurm=False

# Evaluate / render / plot the same sweep
python sweep.py name=ma_board_scans_binary_aiide mode=eval slurm=False
python sweep.py name=ma_board_scans_binary_aiide mode=enjoy slurm=False
python sweep.py name=ma_board_scans_binary_aiide mode=plot slurm=False

# Collect the results into tables and figures under cross_eval/
python cross_eval.py name=ma_board_scans_binary_aiide
```

`mode` is one of `train`, `eval`, `eval_cp` (evaluate over a range of change percentages), `enjoy`, `plot` or
`get_traces`. With `slurm=True` (the default), each config is submitted as a SLURM job through
[submitit](https://github.com/facebookincubator/submitit). The SLURM account comes from `SLURM_ACCOUNT` in a `.env`
file.

If you leave out `name`, `sweep.py` uses the `hypers` defined in [`conf/config_sweeps.py`](conf/config_sweeps.py) and
writes them out to `conf/sweeps/` first.

Sweeps prefixed with `ma_` are the multi-agent experiments (AIIDE 2025). The others (`arf_*`, `obss_*`, `cp_*`,
`act_shape_*`, `diff_size_*`, ...) are the single-agent scaling and generalization experiments (CoG 2024).

## Repository layout

```
├── train.py / train_ma.py      # PPO training: single-agent / multi-agent
├── eval.py / eval_ma.py        # Evaluation: single-agent / multi-agent
├── enjoy.py / enjoy_ma.py      # Render episodes of trained models to GIFs
├── sweep.py                    # Launch train/eval/enjoy/plot jobs over a sweep (locally or on SLURM)
├── cross_eval.py               # Aggregate sweep results into tables and plots
├── eval_change_pct.py          # Evaluation across change percentages
├── eval_ctrls.py               # Evaluation of controllability
├── get_traces.py               # Dump generation traces (states/actions) of trained agents
├── profile_env.py              # Measure environment FPS vs. number of envs (writes results/profile/)
├── models.py                   # Policy / value networks
├── utils.py, utils_ma.py       # Config initialization, env/network construction, helpers
├── conf/
│   ├── config.py               # All Hydra config dataclasses
│   ├── config_sweeps.py        # Python-defined sweeps
│   ├── sweeps/                 # YAML-defined sweeps
│   └── *_hid_params.json       # Hidden sizes that keep #params ~equal across observation sizes
├── envs/
│   ├── pcgrl_env.py            # The core PCGRL environment
│   ├── probs/                  # Problems: binary, maze, dungeon, ...
│   └── reps/                   # Representations: narrow, turtle, wide, nca, ...
├── marl/                       # Multi-agent environment wrappers and models
├── purejaxrl/                  # Vendored PureJaxRL (PPO + wrappers), see purejaxrl/README.md
├── user_defined_freezies/      # Hand-made eval maps with frozen tiles
├── webapp/                     # Flask demo for interacting with trained generators
├── experimental/               # Unmaintained prototypes, see experimental/README.md
├── docs/                       # Notes and experiment plans
├── results/                    # Small result files kept under version control
└── scripts/hpc/                # Helpers for interactive SLURM sessions
```

[`experimental/`](experimental) holds prototypes that the main pipeline doesn't use: alternative interactive viewers,
evolutionary/search baselines and one-off utilities. Some of them are out of date. Run them from the repository root
with `PYTHONPATH=. python experimental/<script>.py`.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{earle2024scaling,
  author    = {Earle, Sam and Jiang, Zehua and Togelius, Julian},
  title     = {Scaling, Control and Generalization in Reinforcement Learning Level Generators},
  booktitle = {2024 IEEE Conference on Games (CoG)},
  year      = {2024},
  pages     = {1--8},
  doi       = {10.1109/CoG60054.2024.10645598}
}

@article{earle2025multiagentpcgrl,
  author  = {Earle, Sam and Jiang, Zehua and Vinitsky, Eugene and Togelius, Julian},
  title   = {Video Game Level Design as a Multi-Agent Reinforcement Learning Problem},
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment},
  volume  = {21},
  number  = {1},
  pages   = {32--42},
  year    = {2025},
  month   = {Nov.},
  doi     = {10.1609/aiide.v21i1.36807},
  url     = {https://ojs.aaai.org/index.php/AIIDE/article/view/36807}
}
```

## Acknowledgements

The RL training code is based on [PureJaxRL](https://github.com/luchris429/purejaxrl) by Chris Lu.

## License

[Apache 2.0](LICENSE)
