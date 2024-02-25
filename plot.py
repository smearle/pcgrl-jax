import json
import os

import hydra
from matplotlib import pyplot as plt
import pandas as pd

from utils import get_exp_dir, init_config

# Get path of current file's directory
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

@hydra.main(config_path="./", config_name="config")
def main(cfg):
    cfg = init_config(cfg)
    exp_dir = cfg.exp_dir
    # Open progress.csv and plot
    progress_csv = os.path.join(__location__, exp_dir, "progress.csv")
    with open(progress_csv, "r") as f:
        df = pd.read_csv(f)
        # Save plot
        plot_path = os.path.join(__location__, exp_dir, "progress.png")
        print(f"Saving plot to {plot_path}")
        # df.plot(x="timestep", y="ep_return", title="Training Progress").get_figure().savefig(plot_path)
        # Plot points instead of a line
        df.plot.scatter(x="timestep", y="ep_return", title="Training Progress").get_figure().savefig(plot_path)
    
    last_timestep = int(df["timestep"].iloc[-1])
    stats = {
        "n_timesteps_trained": last_timestep,
    }
    # Save stats to file
    stats_path = os.path.join(__location__, exp_dir, "misc_stats.json")
    print(f"Saving stats to {stats_path}")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)

    plt.close()

if __name__ == "__main__":
    import sched, time

    main()

    def do_something(scheduler): 
        scheduler.enter(60, 1, do_something, (scheduler,))
        print('Plotting...')
        main()

    my_scheduler = sched.scheduler(time.time, time.sleep)
    my_scheduler.enter(60, 1, do_something, (my_scheduler,))
    my_scheduler.run()