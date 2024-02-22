
import json
import os
from typing import Iterable

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import EvalConfig, SweepConfig, TrainConfig
from eval_change_pct import EvalData, get_change_pcts
from sweep import get_grid_cfgs, hypers
from utils import init_config


CROSS_EVAL_DIR = 'cross_eval'


def cross_eval_main():
    for grid_hypers in hypers:
        default_config = SweepConfig()
        sweep_configs = get_grid_cfgs(default_config, grid_hypers)
        sweep_configs = [init_config(sc) for sc in sweep_configs]

        # FIXME: This part is messy, we have to assume we ran the eval with the 
        #  default params as defined in the class below. We should probably save
        #  the eval config in the eval directory.
        eval_config = EvalConfig()

        name = grid_hypers.pop('NAME') if 'NAME' in grid_hypers else 'default'
        if 'eval_map_width' in grid_hypers:      # if we are sweeping over eval_map_width
            cross_eval_diff_size(name=name, sweep_configs=sweep_configs,
                          eval_config=eval_config, hypers=grid_hypers)
        else:
            cross_eval_misc(name=name, sweep_configs=sweep_configs,
                            eval_config=eval_config, hypers=grid_hypers)
            cross_eval_basic(name=name, sweep_configs=sweep_configs,
                            eval_config=eval_config, hypers=grid_hypers)
            
            if name.startswith('cp_'):
                cross_eval_cp(sweep_name=name, sweep_configs=sweep_configs,
                            eval_config=eval_config)


# Function to bold the maximum value in a column for LaTeX
def format_num(s):
    # Return if not a number
    if not np.issubdtype(s.dtype, np.number):
        return s
    s_max = s.max()

    col = []

    for v in s:
        v_frmt = f'{v:.2f}'
        if v == s_max:
            v_frmt = f'\\textbf{{{v_frmt}}}'
        col.append(v_frmt)
    
    return col


def clean_df_strings(df):
    # Function to replace underscores with spaces in a string
    def replace_underscores(s):
        return s.replace('_', ' ')

    # Replace underscores in index names
    if df.index.names:
        df.index.names = [replace_underscores(name) if name is not None else None for name in df.index.names]

    # Replace underscores in index labels for each level of the MultiIndex
    # for level in range(df.index.nlevels):
    #     df.index = df.index.set_levels([df.index.levels[level].map(replace_underscores)], level=level)

    # Replace underscores in column names
    df.columns = df.columns.map(replace_underscores)

    return df


def cross_eval_basic(name: str, sweep_configs: Iterable[SweepConfig],
                    eval_config: EvalConfig, hypers):

    row_headers = [tuple(v) if isinstance(v, list) else v for v in list(hypers.keys())]
    row_indices = []
    row_vals = []

    # Create a dataframe with basic stats for each experiment
    basic_stats_df = {}
    # for exp_dir, stats in basic_stats.items():
    for sc in sweep_configs:
        sc_stats = json.load(open(f'{sc.exp_dir}/stats.json'))
        row_tpl = tuple(getattr(sc, k) for k in row_headers)
        row_tpl = tuple(tuple(v) if isinstance(v, list) else v for v in row_tpl)
        row_indices.append(row_tpl)
        vals = {}
        for k, v in sc_stats.items():
            vals[k] = v
        row_vals.append(vals)
    
    row_index = pd.MultiIndex.from_tuples(row_indices, names=row_headers)
    basic_stats_df = pd.DataFrame(row_vals, index=row_index)
    
    # Save the dataframe to a csv
    os.makedirs(CROSS_EVAL_DIR, exist_ok=True)
    basic_stats_df.to_csv(os.path.join(CROSS_EVAL_DIR,
                                        f"{name}_basic_stats.csv")) 

    # Save to markdown
    with open(os.path.join(CROSS_EVAL_DIR, f"{name}_basic_stats.md"), 'w') as f:
        f.write(basic_stats_df.to_markdown())

    # Save the dataframe as a latex table
    with open(os.path.join(CROSS_EVAL_DIR, f"{name}_basic_stats.tex"), 'w') as f:
        f.write(basic_stats_df.to_latex())

    # Take averages of stats across seeds, keeping the original row indices
    group_row_indices = [col for col in basic_stats_df.index.names if col != 'seed']
    basic_stats_mean_df = basic_stats_df.groupby(group_row_indices).mean()

    # Save the dataframe to a csv
    basic_stats_mean_df.to_csv(os.path.join(CROSS_EVAL_DIR,
                                        f"{name}_basic_stats_mean.csv"))
    
    # Save to markdown
    with open(os.path.join(CROSS_EVAL_DIR, f"{name}_basic_stats_mean.md"), 'w') as f:
        f.write(basic_stats_mean_df.to_markdown())
    
    # Save the dataframe as a latex table
    with open(os.path.join(CROSS_EVAL_DIR, f"{name}_basic_stats_mean.tex"), 'w') as f:
        f.write(basic_stats_mean_df.to_latex())

    # Now, remove all row indices that have the same value across all rows
    levels_to_drop = \
        [level for level in basic_stats_mean_df.index.names if 
         basic_stats_mean_df.index.get_level_values(level).nunique() == 1]
    
    # Drop these rows
    basic_stats_concise_df = basic_stats_mean_df.droplevel(levels_to_drop)

    # Save the dataframe to a csv
    basic_stats_concise_df.to_csv(os.path.join(CROSS_EVAL_DIR,
                                        f"{name}_basic_stats_concise.csv"))

    # Save to markdown
    with open(os.path.join(CROSS_EVAL_DIR, f"{name}_basic_stats_concise.md"), 'w') as f:
        f.write(basic_stats_concise_df.to_markdown())

    basic_stats_concise_df = clean_df_strings(basic_stats_concise_df)

    # Bold the maximum value in each column
    styled_basic_stats_concise_df = basic_stats_concise_df.apply(format_num)

    # Save the dataframe as a latex table
    with open(os.path.join(CROSS_EVAL_DIR, f"{name}_basic_stats_concise.tex"), 'w') as f:
        f.write(styled_basic_stats_concise_df.to_latex())



        
def cross_eval_misc(name: str, sweep_configs: Iterable[SweepConfig],
                    eval_config: EvalConfig, hypers):

    # Create a dataframe with miscellaneous stats for each experiment
    row_headers = list(hypers.keys())
    row_indices = []
    row_vals = []

    # Create a list of lists to show curves of metrics (e.g. reward) over the 
    # course of training (i.e. as would be logged by tensorboard)
    row_vals_curves = []
    all_timesteps = []

    for sc in sweep_configs:
        exp_dir = sc.exp_dir
        sc_stats = json.load(open(f'{exp_dir}/misc_stats.json'))

        row_tpl = tuple(getattr(sc, k) for k in row_headers)
        row_tpl = tuple(tuple(v) if isinstance(v, list) else v for v in row_tpl)
        row_indices.append(row_tpl)
        
        vals = {}
        for k, v in sc_stats.items():
            vals[k] = v

        row_vals.append(vals)
        
        # Load the `progress.csv`
        train_metrics = pd.read_csv(f'{exp_dir}/progress.csv')
        train_metrics = train_metrics.sort_values(by='timestep', ascending=True)

        ep_returns = train_metrics['ep_return']
        row_vals_curves.append(ep_returns)
        sc_timesteps = train_metrics['timestep']
        all_timesteps.append(sc_timesteps)

    row_index = pd.MultiIndex.from_tuples(row_indices, names=row_headers)
    misc_stats_df = pd.DataFrame(row_vals, index=row_index)

    # Save the dataframe to a csv
    os.makedirs(os.path.join(CROSS_EVAL_DIR, name), exist_ok=True)
    misc_stats_df.to_csv(os.path.join(CROSS_EVAL_DIR, name,
                                        "misc_stats.csv")) 

    # Save to markdown
    with open(os.path.join(CROSS_EVAL_DIR, name, "misc_stats.md"), 'w') as f:
        f.write(misc_stats_df.to_markdown())

    # Save the dataframe as a latex table
    with open(os.path.join(CROSS_EVAL_DIR, name, "misc_stats.tex"), 'w') as f:
        f.write(misc_stats_df.to_latex())

    # Take averages of stats across seeds, keeping the original row indices
    group_row_indices = [col for col in misc_stats_df.index.names if col != 'seed']
    misc_stats_mean_df = misc_stats_df.groupby(group_row_indices).mean()

    # Save the dataframe to a csv
    misc_stats_mean_df.to_csv(os.path.join(CROSS_EVAL_DIR,
                                        name, "misc_stats_mean.csv"))
    
    # Save to markdown
    with open(os.path.join(CROSS_EVAL_DIR, name, "misc_stats_mean.md"), 'w') as f:
        f.write(misc_stats_mean_df.to_markdown())
    
    # Save the dataframe as a latex table
    with open(os.path.join(CROSS_EVAL_DIR, name, "misc_stats_mean.tex"), 'w') as f:
        f.write(misc_stats_mean_df.to_latex())

    # Now, remove all row indices that have the same value across all rows
    levels_to_drop = \
        [level for level in misc_stats_mean_df.index.names if 
         misc_stats_mean_df.index.get_level_values(level).nunique() == 1]
    
    # Drop these rows
    misc_stats_concise_df = misc_stats_mean_df.droplevel(levels_to_drop)

    # Save the dataframe to a csv
    misc_stats_concise_df.to_csv(os.path.join(CROSS_EVAL_DIR,
                                        name, "misc_stats_concise.csv"))

    # Save to markdown
    with open(os.path.join(CROSS_EVAL_DIR, name, "misc_stats_concise.md"), 'w') as f:
        f.write(misc_stats_concise_df.to_markdown())

    misc_stats_concise_df = clean_df_strings(misc_stats_concise_df)

    # Bold the maximum value in each column
    styled_misc_stats_concise_df = misc_stats_concise_df.apply(format_num)

    # Save the dataframe as a latex table
    with open(os.path.join(CROSS_EVAL_DIR, name, "misc_stats_concise.tex"), 'w') as f:
        f.write(styled_misc_stats_concise_df.to_latex())


    # Instead of padding, interpolate the episode returns
    def interpolate_returns(ep_returns, timesteps, all_timesteps):
        # Create a Series with the index set to the timesteps of the ep_returns

        # Wherever there are duplicate `timesteps` values, take the average of the corresponding `ep_returns` values
        ep_returns = ep_returns.groupby(timesteps).mean()
        timesteps = np.unique(timesteps)

        indexed_returns = pd.Series(ep_returns.values, index=timesteps)
        # Reindex the series to the full range of timesteps, interpolating missing values
        interpolated_returns = indexed_returns.reindex(all_timesteps).interpolate(method='linear', limit_direction='forward', axis=0)
        return interpolated_returns

    all_timesteps = np.sort(np.unique(np.concatenate(all_timesteps)))

    row_vals_curves = []
    for sc in sweep_configs:
        exp_dir = sc.exp_dir
        train_metrics = pd.read_csv(f'{exp_dir}/progress.csv')
        train_metrics = train_metrics.sort_values(by='timestep', ascending=True)
        ep_returns = train_metrics['ep_return']
        sc_timesteps = train_metrics['timestep']
        interpolated_returns = interpolate_returns(ep_returns, sc_timesteps, all_timesteps)
        row_vals_curves.append(interpolated_returns)

    # Now, each element in row_vals_curves is a Series of interpolated returns
    metric_curves_df = pd.DataFrame({i: vals for i, vals in enumerate(row_vals_curves)}).T
    metric_curves_df.columns = all_timesteps
    metric_curves_df.index = row_index
    metric_curves_mean = metric_curves_df.groupby(group_row_indices).mean()
    metric_curves_mean = metric_curves_mean.droplevel(levels_to_drop)

    # Create a line plot of the metric curves w.r.t. timesteps. Each row in the
    # column corresponds to a different line
    fig, ax = plt.subplots(figsize=(10, 12))
    for i, row in metric_curves_df.iterrows():
        ax.plot(row, label=str(i))
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Return')
    ax.legend()
    plt.savefig(os.path.join(CROSS_EVAL_DIR, name, "metric_curves.png"))

    fig, ax = plt.subplots()
    for i, row in metric_curves_mean.iterrows():
        ax.plot(row, label=str(i))
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Return')

    # Can manually set these bounds to tweak the visualization
    ax.set_ylim(0, 1.1 * metric_curves_mean.max().max())

    ax.legend()
    plt.savefig(os.path.join(CROSS_EVAL_DIR, name, "metric_curves_mean.png"))


def cross_eval_cp(sweep_name: str, sweep_configs: Iterable[SweepConfig],
                  eval_config: EvalConfig):
    cp_stats = {}
    for sc in sweep_configs:
        log_dir = sc.exp_dir
        sc_cp_stats = EvalData(**json.load(open(f'{log_dir}/cp_stats.json')))
        cp_stats[sc.exp_dir] = (sc_cp_stats, sc)
    
    # Save the dataframe to a csv
    os.makedirs(CROSS_EVAL_DIR, exist_ok=True)

    # Treat change_percentage during training as the independent variable
    cps_to_rews = {}
    for exp_dir, (sc_cp_stats, sc) in cp_stats.items():
        print(f'Gathering stats for experiment: {exp_dir}')
        if sc.change_pct not in cps_to_rews:
            cps_to_rews[sc.change_pct] = []
        cps_to_rews[sc.change_pct].append(sc_cp_stats.cell_rewards)

    # Training CPs to eval CPs to mean reward
    cps_to_cps_mean_rews = {k: np.mean(v, 0) for k, v in cps_to_rews.items()}

    # Note that `n_bins` needs to be the same as it was during eval.
    change_pcts_eval = get_change_pcts(eval_config.n_bins)
    change_pcts_train = sorted(list(cps_to_cps_mean_rews.keys()))

    # Generate a heatmap of change_pct vs eval_cp
    heatmap = np.zeros((len(change_pcts_train), len(change_pcts_eval)))
    for i, cp_train in enumerate(change_pcts_train):
        for j, cp_eval in enumerate(change_pcts_eval):
            heatmap[i, j] = cps_to_cps_mean_rews[cp_train][j]
    
    # Plot heatmap
    fig, ax = plt.subplots()
    im = ax.imshow(heatmap)
    ax.set_xticks(np.arange(len(change_pcts_eval)))
    x_labels = [f'{cp:.2f}' for cp in change_pcts_eval]
    if x_labels[-1] == -1:
        x_labels[-1] = 'Unlimited'
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.arange(len(change_pcts_train)))
    y_labels = [f'{cp:.2f}' for cp in change_pcts_train]
    if y_labels[-1] == -1:
        y_labels[-1] = 'Unlimited'
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Eval Change Percentage')
    ax.set_ylabel('Train Change Percentage')
    fig.colorbar(im)
    plt.savefig(os.path.join(CROSS_EVAL_DIR, sweep_name,
                             "cp_heatmap.png"))


def cross_eval_diff_size(name: str, sweep_configs: Iterable[SweepConfig],
                    eval_config: EvalConfig, hypers):

    row_headers = list(hypers.keys())
    row_indices = []
    row_vals = []

    # Create a dataframe with basic stats for each experiment
    basic_stats_df = {}
    # for exp_dir, stats in basic_stats.items():
    for sc in sweep_configs:
        sc_stats = json.load(open(f'{sc.exp_dir}/eval_map_size_{sc.eval_map_width}_loss_stats.json'))
        row_tpl = tuple(getattr(sc, k) for k in row_headers)
        row_indices.append(row_tpl)
        vals = {}
        for k, v in sc_stats.items():
            vals[k] = v
        row_vals.append(vals)
    
    row_index = pd.MultiIndex.from_tuples(row_indices, names=row_headers)
    basic_stats_df = pd.DataFrame(row_vals, index=row_index)
    
    # Save the dataframe to a csv
    os.makedirs(CROSS_EVAL_DIR, exist_ok=True)
    basic_stats_df.to_csv(os.path.join(CROSS_EVAL_DIR,
                                        name, f"diff_size_size_{eval_config.eval_map_width}_stats.csv")) 

    # Save the dataframe as a latex table
    with open(os.path.join(CROSS_EVAL_DIR, name, f"diff_size_size_{eval_config.eval_map_width}_stats.tex"), 'w') as f:
        f.write(basic_stats_df.to_latex())

    # Take averages of stats across seeds, keeping the original row indices
    group_row_indices = [col for col in basic_stats_df.index.names if col != 'seed']
    basic_stats_mean_df = basic_stats_df.groupby(group_row_indices).mean()

    # Save the dataframe to a csv
    basic_stats_mean_df.to_csv(os.path.join(CROSS_EVAL_DIR,
                                        name, f"diff_size_stats_size_{eval_config.eval_map_width}_mean.csv"))
    
    # Save the dataframe as a latex table
    with open(os.path.join(CROSS_EVAL_DIR, name, f"diff_size_stats_size_{eval_config.eval_map_width}_mean.tex"), 'w') as f:
        f.write(basic_stats_mean_df.to_latex())

    # Now, remove all row indices that have the same value across all rows
    levels_to_drop = \
        [level for level in basic_stats_mean_df.index.names if 
         basic_stats_mean_df.index.get_level_values(level).nunique() == 1]
    
    # Drop these rows
    basic_stats_concise_df = basic_stats_mean_df.droplevel(levels_to_drop)

    # Save the dataframe to a csv
    basic_stats_concise_df.to_csv(os.path.join(CROSS_EVAL_DIR,
                                        name, f"diff_size_stats_size_{eval_config.eval_map_width}_concise.csv"))

    basic_stats_concise_df = clean_df_strings(basic_stats_concise_df)

    # Bold the maximum value in each column
    styled_basic_stats_concise_df = basic_stats_concise_df.apply(format_num)

    # Save the dataframe as a latex table
    with open(os.path.join(CROSS_EVAL_DIR, name, f"diff_size_stats_size_{eval_config.eval_map_width}_concise.tex"), 'w') as f:
        f.write(styled_basic_stats_concise_df.to_latex())



if __name__ == '__main__':
    cross_eval_main()