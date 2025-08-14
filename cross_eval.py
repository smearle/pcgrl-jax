
import copy
from itertools import product
import json
import os
from typing import Iterable

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import wandb
import yaml
from scipy import stats

from conf.config import EvalConfig, EvalMultiAgentConfig, SweepConfig, TrainConfig
from eval import get_eval_name
from eval_change_pct import EvalData, get_change_pcts
from ma_utils import ma_init_config
from sweep import get_grid_cfgs, eval_hypers, get_sweep_cfgs
from utils import get_sweep_conf_path, init_config, load_sweep_hypers, write_sweep_confs


CROSS_EVAL_DIR = 'cross_eval'

# The index of the metric in the column MultiIndex. When 0, the metric will go on top. (This is good when we are 
# are sweeping over eval metrics and want to show only one metric to save space.) Otherwise, should be -1.
METRIC_COL_TPL_IDX = 0

table_name_remaps = {
    'min_min_ep_loss': 'min. loss',
    'mean_min_ep_loss': 'mean loss',
    'max_board_scans': 'max. board scans',
    'eval_randomize_map_shape': 'rand. map shape',
    'randomize_map_shape': 'rand. map shape',
    'eval map width': 'map width',
}

@hydra.main(version_base="1.3", config_path='./', config_name='batch_pcgrl')
def cross_eval_main(cfg: SweepConfig):
    sweep_conf_path = get_sweep_conf_path(cfg)
    sweep_conf_exists = os.path.exists(sweep_conf_path)

    if cfg.name is not None:
        _hypers, _eval_hypers = load_sweep_hypers(cfg)
        _hypers = [_hypers]
    else:
        from conf.config_sweeps import hypers
        _hypers = hypers
        _eval_hypers = eval_hypers
        write_sweep_confs(_hypers, _eval_hypers)
    
    sweep_grid(cfg, _hypers, _eval_hypers)


def get_sweep_configs(default_cfg, grid_hypers, _eval_hypers, mode):
    sweep_configs = get_sweep_cfgs(default_cfg, grid_hypers, mode=mode, eval_hypers=_eval_hypers)
    sweep_configs = [OmegaConf.create(sc) for sc in sweep_configs]
    sweep_configs = [init_config(sc) for sc in sweep_configs]

    for sc in sweep_configs:
        for k, v in sc.__dict__.items():
            if k.startswith('obs_size'):
                if v == -1:
                    setattr(sc, k, sc.map_width * 2 - 1)

    return sweep_configs


def sweep_grid(cfg: SweepConfig, grid_hypers, _eval_hypers):
    if grid_hypers[0].get('multiagent', False):
        default_cfg = EvalMultiAgentConfig(multiagent=True)
    else:
        default_cfg = EvalConfig()

    train_sweep_configs = get_sweep_configs(default_cfg, grid_hypers, _eval_hypers, mode='train')
    eval_sweep_configs = get_sweep_configs(default_cfg, grid_hypers, _eval_hypers, mode='eval')

    # FIXME: This part is messy, we have to assume we ran the eval with the 
    #  default params as defined in the class below. We should probably save
    #  the eval config in the eval directory.
    eval_config = EvalConfig()
    init_config(eval_config)

    name = grid_hypers[0].get('NAME', 'default')
    [gh.pop('NAME') for gh in grid_hypers]

    # Save the eval config to a yaml at `conf/sweeps/{name}.yaml`        

    if 'eval_map_width' in grid_hypers:      # if we are sweeping over eval_map_width
        cross_eval_diff_size(name=name, sweep_configs=eval_sweep_configs,
                        eval_config=eval_config, hypers=grid_hypers)
    else:
        os.makedirs(os.path.join(CROSS_EVAL_DIR, name), exist_ok=True)
        cross_eval_misc(name=name, sweep_configs=train_sweep_configs,
                        eval_config=eval_config, hypers=grid_hypers)
        cross_eval_basic(name=name, sweep_configs=eval_sweep_configs,
                        eval_config=eval_config, hypers=grid_hypers, eval_hypers=_eval_hypers)
        
        if name.startswith('cp_'):
            cross_eval_cp(sweep_name=name, sweep_configs=sweep_configs,
                        eval_config=eval_config)


# Function to bold the maximum value in a column for LaTeX
def format_num(s):
    # Return if not a number
    if not np.issubdtype(s.dtype, np.number):
        return s
    is_pct = False
    # Check if the header of the row
    if is_loss_column(s.name):
        is_pct = True
        s_best = s.min()

    else:
        s_best = s.max()

    col = []

    for v in s:
        if is_pct:
            v_frmt = f'{v:.2%}'
            v_frmt = v_frmt.replace('%', '\\%')
        else:
            v_frmt = f'{v:.2f}'
        if v == s_best:
            v_frmt = f'\\textbf{{{v_frmt}}}'
        col.append(v_frmt)
    
    return col


def is_loss_column(col):
    if isinstance(col, str) and 'loss' in col:
        return True
    elif isinstance(col, tuple) and 'loss' in col[METRIC_COL_TPL_IDX]:
        return True
    return False


def replace_underscores(s):
    return s.replace('_', ' ')


def process_col_str(s):
    if isinstance(s, str):
        if s in table_name_remaps:
            s = table_name_remaps[s]
        else:
            s = replace_underscores(s)
    return s


# Function to replace underscores with spaces in a string
def process_col_tpls(t):
    if isinstance(t, str):
        return process_col_str(t)
    new_s = []
    for s in t:
        s = process_col_str(s)
        new_s.append(s)
    return tuple(new_s)


def clean_df_strings(df):

    # Replace underscores in index names
    if df.index.names:
        # df.index.names = [replace_underscores(name) if name is not None else None for name in df.index.names]
        new_names = []
        for name in df.index.names:
            if name is None:
                continue
            if name in table_name_remaps:
                new_names.append(table_name_remaps[name])
            else:
                new_names.append(replace_underscores(name))
        df.index.names = new_names

    if df.columns.names:
        # df.columns.names = [replace_underscores(name) if name is not None else None for name in df.columns.names]
        new_names = []
        for name in df.columns.names:
            if name is None:
                new_names.append(name)
            elif name in table_name_remaps:
                new_names.append(table_name_remaps[name])
            else:
                new_names.append(replace_underscores(name))
        df.columns.names = new_names
    
    for i, tpl in enumerate(df.index.values):
        if not isinstance(tpl, tuple):
            continue
        tpl = (str(t) for t in tpl)
        df.index.values[i] = tuple(tpl)

    # Replace underscores in index labels for each level of the MultiIndex
    # for level in range(df.index.nlevels):
    #     df.index = df.index.set_levels([df.index.levels[level].map(replace_underscores)], level=level)

    # Replace underscores in column names
    df.columns = df.columns.map(process_col_tpls)

    return df


def cross_eval_basic(name: str, sweep_configs: Iterable[SweepConfig],
                    eval_config: EvalConfig, hypers, eval_hypers):

    # Save the eval hypers to the cross_eval directory, so that we know of any special eval hyperparameters that were
    # applied during eval.
    # with open(os.path.join(CROSS_EVAL_DIR, name, "eval_hypers.yaml"), 'w') as f:
    #     yaml.dump(eval_hypers, f)

    eval_hyper_ks = [k for k in eval_hypers]
    eval_hyper_combos = list(product(*[eval_hypers[k] for k in eval_hypers]))

    eval_sweep_name = ('eval_' + '_'.join(k.strip('eval_') for k, v in eval_hypers.items() if len(v) > 1 and k != 'metrics_to_keep') if 
                        len(eval_hypers) > 0 else '')

    _metrics_to_keep = None
    if 'eval_map_width' in eval_hyper_ks:
        _metrics_to_keep = eval_config.metrics_to_keep

    col_headers = [k for k in eval_hyper_ks]
    col_headers.insert(METRIC_COL_TPL_IDX, '')
    col_indices = set({})

    row_headers = []
    row_headers = [tuple(v) if isinstance(v, list) else v for v in list(hypers[0].keys())]
    row_indices = []
    row_vals = []

    # Create a dataframe with basic stats for each experiment
    basic_stats_df = {}
    # for exp_dir, stats in basic_stats.items():
    for sc in sweep_configs:

        sweep_eval_configs = []

        # Do this so that we can get the correct stats file depending on eval parameters
        # sc = init_config_for_eval(sc)

        # For each train config, also sweep over eval params to get all the relevant stats
        for eval_hyper_combo in eval_hyper_combos:
            new_sec = copy.deepcopy(sc)
            for k, v in zip(eval_hyper_ks, eval_hyper_combo):
                setattr(new_sec, k, v)
            sweep_eval_configs.append(new_sec)
        
        row_tpl = tuple(getattr(sc, k) for k in row_headers)
        row_tpl = tuple(tuple(v) if isinstance(v, list) else v for v in row_tpl)
        row_indices.append(row_tpl)
        
        vals = {}
        for sec in sweep_eval_configs:
            sec_col_tpl = [getattr(sec, k) for k in eval_hyper_ks]
            print(f"Collecting eval metrics from: {sc.exp_dir}")
            sc_stats = json.load(open(
                os.path.join(f'{sc.exp_dir}', 
                            'stats' + get_eval_name(sec, sc) + '.json')))
            for k, v in sc_stats.items():
                col_tpl = copy.deepcopy(sec_col_tpl)
                col_tpl.insert(METRIC_COL_TPL_IDX, k)
                col_tpl = tuple(col_tpl)
                col_indices.add(col_tpl)

                # FIXME: HACK reward to be fair among experiments with different numbers of agents. This should be handles in the eval script or the LogWrapper.
                if k == 'mean_ep_reward':
                    v = v / sec.n_agents
                vals[col_tpl] = v
        row_vals.append(vals)

    col_index = pd.MultiIndex.from_tuples(col_indices, names=col_headers)
    row_index = pd.MultiIndex.from_tuples(row_indices, names=row_headers)
    basic_stats_df = pd.DataFrame(row_vals, index=row_index, columns=col_index)

    # Sort columns
    basic_stats_df = basic_stats_df.sort_index(axis=1)
    
    # Save the dataframe to a csv
    # os.makedirs(CROSS_EVAL_DIR, exist_ok=True)
    # basic_stats_df.to_csv(os.path.join(CROSS_EVAL_DIR, name,
    #                                     "basic_stats.csv")) 

    # Save to markdown
    with open(os.path.join(CROSS_EVAL_DIR, name, "basic_stats.md"), 'w') as f:
        f.write(basic_stats_df.to_markdown())

    # Save the dataframe as a latex table
    # with open(os.path.join(CROSS_EVAL_DIR, name, "basic_stats.tex"), 'w') as f:
    #     f.write(basic_stats_df.to_latex())

    # Step 1: Calculate mean and standard deviation
    group_row_indices = [col for col in basic_stats_df.index.names if col != 'seed']
    basic_stats_mean_df = basic_stats_df.groupby(group_row_indices).mean()
    basic_stats_std_df = basic_stats_df.groupby(group_row_indices).std()

    # Step 2: Create a new DataFrame with the formatted "mean +/- std%" strings
    # Initialize an empty DataFrame with the same index and columns
    meanstd_df = pd.DataFrame(index=basic_stats_mean_df.index, columns=basic_stats_mean_df.columns)

    # Iterate over each cell to format
    for col in basic_stats_mean_df.columns:
        if is_loss_column(col):
            is_pct = True
            m_best = basic_stats_mean_df[col].min()
        else:
            is_pct = False
            m_best = basic_stats_mean_df[col].max()

        for idx in basic_stats_mean_df.index:
            mean = basic_stats_mean_df.at[idx, col]
            std = basic_stats_std_df.at[idx, col]

            if is_pct:
                mean_frmt = f'{mean:.2%}'
                mean_frmt = mean_frmt.replace('%', '\\%')
                std_frmt = f'{std:.2%}'
                std_frmt = std_frmt.replace('%', '\\%')
            else:
                mean_frmt = f'{mean:.2f}'
                std_frmt = f'{std:.2f}'
            if mean == m_best:
                mean_frmt = f'\\textbf{{{mean_frmt}}}'
                std_frmt = f'\\textbf{{{std_frmt}}}'
            meanstd_df.at[idx, col] = f'{mean_frmt} ± {std_frmt}'

    # Note: If you want the std as a percentage of the mean, replace the formatting line with:
    # formatted_df.loc[idx, col] = f"{mean:.2f} +/- {std/mean*100:.2f}%
    basic_stats_mean_df = meanstd_df

    # Save the dataframe to a csv
    # basic_stats_mean_df.to_csv(os.path.join(CROSS_EVAL_DIR, name,
    #                                     "basic_stats_mean.csv"))
    
    # Save to markdown
    with open(os.path.join(CROSS_EVAL_DIR, name, f"{eval_sweep_name}_basic_stats_mean.md"), 'w') as f:
        f.write(basic_stats_mean_df.to_markdown())
    
    # Save the dataframe as a latex table
    # with open(os.path.join(CROSS_EVAL_DIR, name, "basic_stats_mean.tex"), 'w') as f:
    #     f.write(basic_stats_mean_df.to_latex())

    # Now, remove all row indices that have the same value across all rows
    row_levels_to_drop = \
        [level for level in basic_stats_mean_df.index.names if 
         basic_stats_mean_df.index.get_level_values(level).nunique() == 1]

    # Drop these rows
    basic_stats_concise_df = basic_stats_mean_df.droplevel(row_levels_to_drop)

    # Similarly, remove all column indices that have the same value across all columns
    col_levels_to_drop = \
        [level for level in basic_stats_mean_df.columns.names if
            basic_stats_mean_df.columns.get_level_values(level).nunique() == 1]
    
    # Drop these columns
    basic_stats_concise_df = basic_stats_concise_df.droplevel(col_levels_to_drop, axis=1)
    basic_stats_df = basic_stats_df.droplevel(col_levels_to_drop, axis=1)

    # Drop the `n_parameters` `n_eval_eps` metrics, and others if `metrics_to_keep` is specified
    for col_tpl in basic_stats_concise_df.columns:
        if isinstance(col_tpl, str):
            metric_str = col_tpl
        else:
            metric_str = col_tpl[METRIC_COL_TPL_IDX]
        if metric_str == 'n_parameters' or metric_str == 'n_eval_eps':
            basic_stats_concise_df = basic_stats_concise_df.drop(columns=col_tpl)
            basic_stats_df = basic_stats_df.drop(columns=col_tpl)
        elif _metrics_to_keep is not None and metric_str not in _metrics_to_keep:
            basic_stats_concise_df = basic_stats_concise_df.drop(columns=col_tpl)
            basic_stats_df = basic_stats_df.drop(columns=col_tpl)

    # Compute pairwise p-values across groups (e.g., seeds grouped by hyperparameters) using Mann-Whitney U tests.
    def _label_from_key(key, keys):
        if not isinstance(key, tuple):
            key = (key,)
        return ",".join(f"{k}={v}" for k, v in zip(keys, key) if k not in row_levels_to_drop)

    def compute_pairwise_mannwhitney_pvalues(df: pd.DataFrame) -> pd.DataFrame:
        # Group rows by all index levels except 'seed'; each group's distribution is values over seeds
        group_levels = [lvl for lvl in df.index.names if lvl != 'seed']
        if len(group_levels) == 0:
            raise ValueError("Expected a 'seed' level in the index to form distributions per group.")

        results = []
        grouped = df.groupby(group_levels, dropna=False)
        group_keys = list(grouped.groups.keys())
        if len(group_keys) < 2:
            return pd.DataFrame(columns=[
                'metric', 'group_a', 'group_b', 'n_a', 'n_b', 'stat', 'pvalue'
            ])

        # For each column (metric/eval setting), compute pairwise tests across all groups
        for col in df.columns:
            # Collect distributions per group
            group_to_vals = {}
            for gk, gdf in grouped:
                s = gdf[col]
                vals = pd.Series(s).dropna().to_numpy()
                group_to_vals[gk] = vals

            # Pairwise comparisons
            from itertools import combinations
            for g1, g2 in combinations(group_keys, 2):
                a = group_to_vals.get(g1, np.array([]))
                b = group_to_vals.get(g2, np.array([]))
                na, nb = a.size, b.size
                if na == 0 or nb == 0:
                    stat, p = np.nan, np.nan
                else:
                    try:
                        stat, p = stats.mannwhitneyu(a, b, alternative='two-sided', method='auto')
                    except TypeError:
                        stat, p = stats.mannwhitneyu(a, b, alternative='two-sided')

                # Column label to a compact string (preserve metric name at METRIC_COL_TPL_IDX)
                if isinstance(col, tuple):
                    metric_label = col[METRIC_COL_TPL_IDX]
                else:
                    metric_label = col
                results.append({
                    'metric': metric_label,
                    'column': str(col),
                    'group_a': _label_from_key(g1, group_levels),
                    'group_b': _label_from_key(g2, group_levels),
                    'n_a': na,
                    'n_b': nb,
                    'stat': float(stat) if np.isfinite(stat) else np.nan,
                    'pvalue': float(p) if np.isfinite(p) else np.nan,
                })
        return pd.DataFrame(results)

    pvals_df = compute_pairwise_mannwhitney_pvalues(basic_stats_df)
    if not pvals_df.empty:
        pvals_df['significant_0.05'] = pvals_df['pvalue'] <= 0.05
        os.makedirs(os.path.join(CROSS_EVAL_DIR, name), exist_ok=True)
        # Include eval sweep name to disambiguate different eval settings
        eval_sweep_name = ('eval_' + '_'.join(k.strip('eval_') for k, v in eval_hypers.items() if len(v) > 1 and k != 'metrics_to_keep') if 
                            len(eval_hypers) > 0 else '')
        csv_path = os.path.join(CROSS_EVAL_DIR, name, f"{eval_sweep_name}_pairwise_pvalues.csv")
        md_path = os.path.join(CROSS_EVAL_DIR, name, f"{eval_sweep_name}_pairwise_pvalues.md")
        pvals_df.to_csv(csv_path, index=False)
        with open(md_path, 'w') as f:
            f.write(pvals_df.to_markdown(index=False))

    # Save the dataframe to a csv
    # basic_stats_concise_df.to_csv(os.path.join(CROSS_EVAL_DIR,
    #                                     name, "basic_stats_concise.csv"))

    # Save to markdown
    with open(os.path.join(CROSS_EVAL_DIR, name, f"{eval_sweep_name}_basic_stats_concise.md"), 'w') as f:
        f.write(basic_stats_concise_df.to_markdown())

    # Bold the maximum value in each column
    # styled_basic_stats_concise_df = basic_stats_concise_df.apply(format_num)

    styled_basic_stats_concise_df = clean_df_strings(basic_stats_concise_df)

    latex_str = styled_basic_stats_concise_df.to_latex(
        multicolumn_format='c',
    )
    latex_str_lines = latex_str.split('\n')
    # Add `\centering` to the beginning of the table
    latex_str_lines.insert(0, '\\adjustbox{max width=\\textwidth}{%')
    latex_str_lines.insert(0, '\\centering')
    n_col_header_rows = len(styled_basic_stats_concise_df.columns.names)
    i = 4 + n_col_header_rows
    latex_str_lines.insert(i, '\\toprule')
    # Add `\label` to the end of the table
    latex_str_lines.append(f'\\label{{tab:{name}_{eval_sweep_name}}}')
    latex_str_lines.append('}')
    latex_str = '\n'.join(latex_str_lines)

    # Save the dataframe as a latex table
    with open(os.path.join(CROSS_EVAL_DIR, name, f"{name}_{eval_sweep_name}.tex"), 'w') as f:
        f.write(latex_str)

    print(f"Basic stats for {name} saved to {CROSS_EVAL_DIR}/{name}.")

        
def cross_eval_misc(name: str, sweep_configs: Iterable[TrainConfig],

                    eval_config: EvalConfig, hypers):

    # Create a dataframe with miscellaneous stats for each experiment
    row_headers = list(hypers[0].keys())
    row_indices = []
    row_vals = []

    # Create a list of lists to show curves of metrics (e.g. reward) over the 
    # course of training (i.e. as would be logged by tensorboard)
    row_vals_curves = []
    all_timesteps = []

    wandb_api = wandb.Api()

    for sc in sweep_configs:
        if sc.multiagent:
            ma_init_config(sc)
        else:
            init_config(sc)
        exp_dir = sc.exp_dir
        
        # Load the `progress.csv`
        csv_path = os.path.join(exp_dir, 'progress.csv')
        wandb_path = os.path.join(exp_dir, 'wandb_run_id.txt')
        if not (os.path.isfile(csv_path) or os.path.isfile(wandb_path)):
            print(f"Skipping {exp_dir} as it does not have a progress.csv or wandb-run-id.txt file.")
            continue
        if not os.path.isfile(csv_path):
            print(f"Loading wandb run for {sc.exp_dir}...")
            with open(wandb_path, 'r') as f:
                wandb_run_id = f.read()
            try:
                sc_run = wandb_api.run(f'/{EvalMultiAgentConfig.PROJECT}/{wandb_run_id}')
            except wandb.errors.CommError:
                wandb_run_dirs = os.listdir(os.path.join(exp_dir, 'wandb'))
                for d in wandb_run_dirs:
                    if d.startswith(f'run-{wandb_run_id}'):
                        os.system(f'wandb sync {os.path.join(exp_dir, "wandb", d)}')

            train_metrics = sc_run.history()
            if '_step' not in train_metrics:
                breakpoint()
            train_metrics = train_metrics.sort_values(by='_step', ascending=True)
            sc_timesteps = train_metrics['_step'] * sc._num_actors * sc.num_steps
            max_timestep = sc_timesteps.max()
            if 'returns' not in train_metrics:
                breakpoint()
            ep_returns = train_metrics['returns']
        else:
            train_metrics = pd.read_csv(csv_path)
            train_metrics = train_metrics.sort_values(by='timestep', ascending=True)

            # misc_stats_path = os.path.join(exp_dir, 'misc_stats.json')
            # if os.path.exists(misc_stats_path):
            #     sc_stats = json.load(open(f'{exp_dir}/misc_stats.json'))
            # else:
            max_timestep = train_metrics['timestep'].max()

            ep_returns = train_metrics['ep_return']
            sc_timesteps = train_metrics['timestep']

        row_vals_curves.append(ep_returns)
        all_timesteps.append(sc_timesteps)

        sc_stats = {'n_timesteps_trained': max_timestep}

        row_tpl = tuple(getattr(sc, k) for k in row_headers)
        row_tpl = tuple(tuple(v) if isinstance(v, list) else v for v in row_tpl)
        row_indices.append(row_tpl)
        
        vals = {}
        for k, v in sc_stats.items():
            vals[k] = v

        row_vals.append(vals)        

    row_index = pd.MultiIndex.from_tuples(row_indices, names=row_headers)
    misc_stats_df = pd.DataFrame(row_vals, index=row_index)

    # Save the dataframe to a csv
    # misc_stats_df.to_csv(os.path.join(CROSS_EVAL_DIR, name,
    #                                     "misc_stats.csv")) 

    # Save to markdown
    with open(os.path.join(CROSS_EVAL_DIR, name, "misc_stats.md"), 'w') as f:
        f.write(misc_stats_df.to_markdown())

    # Save the dataframe as a latex table
    # with open(os.path.join(CROSS_EVAL_DIR, name, "misc_stats.tex"), 'w') as f:
    #     f.write(misc_stats_df.to_latex())

    # Take averages of stats across seeds, keeping the original row indices
    group_row_indices = [col for col in misc_stats_df.index.names if col != 'seed']
    misc_stats_mean_df = misc_stats_df.groupby(group_row_indices).mean()

    # Save the dataframe to a csv
    # misc_stats_mean_df.to_csv(os.path.join(CROSS_EVAL_DIR,
    #                                     name, "misc_stats_mean.csv"))
    
    # Save to markdown
    with open(os.path.join(CROSS_EVAL_DIR, name, "misc_stats_mean.md"), 'w') as f:
        f.write(misc_stats_mean_df.to_markdown())
    
    # Save the dataframe as a latex table
    # with open(os.path.join(CROSS_EVAL_DIR, name, "misc_stats_mean.tex"), 'w') as f:
    #     f.write(misc_stats_mean_df.to_latex())

    # Now, remove all row indices that have the same value across all rows
    levels_to_drop = \
       [level for level in misc_stats_mean_df.index.names if 
         misc_stats_mean_df.index.get_level_values(level).nunique() == 1]
    levels_to_keep = \
        [level for level in misc_stats_mean_df.index.names if
            misc_stats_mean_df.index.get_level_values(level).nunique() > 1]
    
    # Drop these rows
    misc_stats_concise_df = misc_stats_mean_df.droplevel(levels_to_drop)

    # Save the dataframe to a csv
    # misc_stats_concise_df.to_csv(os.path.join(CROSS_EVAL_DIR,
    #                                     name, "misc_stats_concise.csv"))

    # Save to markdown
    with open(os.path.join(CROSS_EVAL_DIR, name, "misc_stats_concise.md"), 'w') as f:
        f.write(misc_stats_concise_df.to_markdown())

    misc_stats_concise_df = clean_df_strings(misc_stats_concise_df)

    # Bold the maximum value in each column
    styled_misc_stats_concise_df = misc_stats_concise_df.apply(format_num)

    # Save the dataframe as a latex table
    with open(os.path.join(CROSS_EVAL_DIR, name, "misc_stats_concise.tex"), 'w') as f:
        f.write(styled_misc_stats_concise_df.to_latex())


    def interpolate_returns(ep_returns, timesteps, all_timesteps):
        # Group by timesteps and take the mean for duplicate values
        ep_returns = pd.Series(ep_returns).groupby(timesteps).mean()
        timesteps = np.unique(timesteps)
        timesteps = timesteps[~np.isnan(timesteps)]
        
        # Create a Series with the index set to the unique timesteps of the ep_returns
        indexed_returns = pd.Series(ep_returns.values, index=timesteps)
        
        # Reindex the series to include all timesteps, introducing NaNs for missing values
        indexed_returns = indexed_returns.reindex(all_timesteps)
        
        # Interpolate missing values, ensuring forward fill to handle right edge
        interpolated_returns = indexed_returns.interpolate(method='linear', limit_direction='backward', axis=0)
        
        return interpolated_returns

    all_timesteps = np.sort(np.unique(np.concatenate(all_timesteps)))

    row_vals_curves = []
    for i, sc in enumerate(sweep_configs):
        if hasattr(sc, "obs_size"):
            if sc.obs_size == -1:
                if eval_config.eval_map_width is not None:
                    mw = eval_config.eval_map_width
                else:
                    mw = eval_config.map_width
                # Why exactly is this necessary? And should we really be inheriting from *eval* map width?
                sc.obs_size = mw * 2 - 1
        exp_dir = sc.exp_dir
        csv_path = os.path.join(exp_dir, 'progress.csv')
        wandb_path = os.path.join(exp_dir, 'wandb_run_id.txt')
        if not (os.path.isfile(csv_path) or os.path.isfile(wandb_path)):
            print(f"Skipping {sc.exp_dir} because it has no progress.csv or wandb_run_id.txt.")
            continue
        if not os.path.isfile(csv_path):
            with open(wandb_path, 'r') as f:
                wandb_run_id = f.read()
            try:
                sc_run = wandb_api.run(f'/{EvalMultiAgentConfig.PROJECT}/{wandb_run_id}')
            except wandb.errors.CommError:
                wandb_run_dirs = os.listdir(os.path.join(exp_dir, 'wandb'))
                for d in wandb_run_dirs:
                    os.system(f'wandb sync {os.path.join(exp_dir, "wandb", d)}')
            train_metrics = sc_run.history()
            train_metrics = train_metrics.sort_values(by='_step', ascending=True)
            sc_timesteps = train_metrics['_step'] * sc._num_actors * sc.num_steps
            ep_returns = train_metrics['returns']

        else:
            train_metrics = pd.read_csv(csv_path)
            train_metrics = train_metrics.sort_values(by='timestep', ascending=True)
            
            ep_returns = train_metrics['ep_return']
            sc_timesteps = train_metrics['timestep']
        if sc_timesteps.shape[0] == 0:
            print(f"Skipping {sc.exp_dir} because it has no timesteps.")
            continue
        interpolated_returns = interpolate_returns(ep_returns, sc_timesteps, all_timesteps)
        row_vals_curves.append(interpolated_returns)

    # Now, each element in row_vals_curves is a Series of interpolated returns
    metric_curves_df = pd.DataFrame({i: vals for i, vals in enumerate(row_vals_curves)}).T
    metric_curves_df.columns = all_timesteps
    metric_curves_df.index = row_index
    metric_curves_mean = metric_curves_df.groupby(group_row_indices).mean()
    metric_curves_mean = metric_curves_mean.droplevel(levels_to_drop)
    metric_curves_std = metric_curves_df.groupby(group_row_indices).std()
    metric_curves_std = metric_curves_std.droplevel(levels_to_drop)

    # Create a line plot of the metric curves w.r.t. timesteps. Each row in the
    # column corresponds to a different line
    plt.savefig(os.path.join(CROSS_EVAL_DIR, name, f"metric_curves.png"))

    fig, ax = plt.subplots()
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Return')
    # cut off the first and last 100 timesteps to remove outliers
    metric_curves_mean = metric_curves_mean.drop(columns=metric_curves_mean.columns[:25])
    metric_curves_mean = metric_curves_mean.drop(columns=metric_curves_mean.columns[-25:])
    metric_curves_std = metric_curves_std.drop(columns=metric_curves_std.columns[:25])
    metric_curves_std = metric_curves_std.drop(columns=metric_curves_std.columns[-25:])

    columns = copy.deepcopy(metric_curves_mean.columns)
    # columns = columns[100:-100]
    for ((i, row), (_, row_std)) in zip(metric_curves_mean.iterrows(), metric_curves_std.iterrows()):

        if len(row) == 0:
            continue
        # Apply a convolution to smooth the curve
        row = np.convolve(row, np.ones(10), 'same') / 10
        # row = row[100:-100]
        # row = np.convolve(row, np.ones(10), 'valid') / 10
        # turn it back into a pandas series
        row = pd.Series(row, index=columns)
        
        # drop the first 100 timesteps to remove outliers caused by conv
        if row.index.shape[0] > 100:
            row = row.drop(row.index[:25])
            row = row.drop(row.index[-25:])
            row_std = row_std.drop(row_std.index[:25])
            row_std = row_std.drop(row_std.index[-25:])
        ax.plot(row, label=str(i))
        ax.fill_between(
            row.index,
            row - row_std,
            row + row_std,
            alpha=0.2,
            color=ax.get_lines()[-1].get_color(),
            linewidth=0.5,
            linestyle='--',
        )

    ax.set_ylim(bottom=30, top=(metric_curves_mean + metric_curves_std).max().max())

    metric_curves_mean.columns = columns
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Return')

    # To get the ymin, drop the first timesteps where there tend to be outliers
    if metric_curves_mean.shape[1] > 100:
        ymin = metric_curves_mean.drop(columns=metric_curves_mean.columns[:100]).min().min()
    else:
        ymin = metric_curves_mean.drop(columns=metric_curves_mean.columns).min().min()

    # Can manually set these bounds to tweak the visualization
    # ax.set_ylim(ymin, 1.1 * np.nanmax(metric_curves_mean))

    legend_title = ', '.join(levels_to_keep).replace('_', ' ')
    ax.legend(title=legend_title)
    plt.savefig(os.path.join(CROSS_EVAL_DIR, name, f"{name}_metric_curves_mean.png"))

    print(f"Misc stats for {name} saved to {CROSS_EVAL_DIR}/{name}.")


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
    x_labels = [f'{cp:.1f}' for cp in change_pcts_eval]
    if x_labels[-1] == -1:
        x_labels[-1] = 'Unlimited'
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.arange(len(change_pcts_train)))
    y_labels = [f'{cp:.1f}' for cp in change_pcts_train]
    if y_labels[-1] == -1:
        y_labels[-1] = 'Unlimited'
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Eval Change Percentage')
    ax.set_ylabel('Train Change Percentage')
    fig.colorbar(im)
    plt.savefig(os.path.join(CROSS_EVAL_DIR, sweep_name,
                             f"{sweep_name}_cp_heatmap.png"))


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