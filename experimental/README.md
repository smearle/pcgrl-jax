# Experimental scripts

These are prototypes and one-off tools. The main training/evaluation pipeline doesn't use them, they aren't
maintained, and some of them no longer run against the current codebase.

They import modules from the repository root (`utils`, `train`, `conf`, `envs`, ...), so run them from the root with
the root on `PYTHONPATH`:

```bash
PYTHONPATH=. python experimental/evo_map.py problem=binary
```

| Script | What it is | Status |
|---|---|---|
| `enjoy_cpu.py`, `enjoy_different_size.py`, `enjoy_candy.py` | Variants of `enjoy.py` (CPU-only, different map sizes, candy env) | Loads |
| `enjoy_tk.py`, `enjoy_pygame.py`, `enjoy_pygui.py` | Interactive viewers (Tk / pygame / PySimpleGUI) | Needs `pygame` / `PySimpleGUI` for the last two |
| `enjoy_ctrls.py` | Render controllable agents over metric targets | Broken: `EvalCtrlsConfig` no longer exists |
| `enjoy_ue.py` | Another variant of `enjoy.py` | Broken (deliberately disabled) |
| `gtk_gui.py` | GTK GUI prototype | Broken (syntax error) |
| `train_accel.py`, `evo_accel.py` | ACCEL-style training with evolved levels (`conf/train_accel.yaml`) | Loads |
| `evo_map.py` | Evolve maps directly, no RL | Loads |
| `evolve.py` | EvoJAX-based evolution | Needs `evojax` |
| `illuminate.py`, `qdax.py` | MAP-Elites with QDax | Broken |
| `search.py`, `search_mctx.py` | Search-based baselines over level edits (`search_mctx.py` uses MCTS) | `search_mctx.py` needs `mctx` |
| `nca.py` | Standalone neural cellular automaton (NCA) prototype | Loads |
| `ppo_2.py` | Old PPO variant | Broken |
| `hash.py` | JAX xxhash for hashing maps | Runs |
| `gen_hid_params_per_model_obs_size.py` | Like `gen_hid_params_per_obs_size.py`, keyed per model | Loads |
| `test_dum_map.py`, `test_eval_map.py` | Ad-hoc checks for the eval maps in `user_defined_freezies/` | Loads |

"Loads" means the script imports and builds its config. It hasn't been run end to end.
