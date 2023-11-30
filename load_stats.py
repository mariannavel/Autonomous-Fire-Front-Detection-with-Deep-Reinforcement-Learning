import pickle as pkl
import os
from utils.plots import *

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

NUM_SAMPLES = 100

def load_data(logdir, fname1, fname2):
    with open(os.path.join(logdir, fname1), "rb") as fp:
        train_stats = pkl.load(fp)
    with open(os.path.join(logdir, fname2), "rb") as fp:
        test_stats = pkl.load(fp)
    return train_stats, test_stats

def concatenate_stats(stats1, stats2):
    """
    stats1, stats2: dictionaries of lists
    """
    for key in stats1:
        stats1[key].extend(stats2[key])
    return stats1


def produce_figures(logdir_rand, logdir_strat):
    train_rand, _ = load_data(logdir_rand,"train_stats_E20", "test_stats_E20")
    train_strat, _ = load_data(logdir_strat, "train512_epoch2000", "test512_epoch2000")

    plot2curves(train_rand["loss"], train_rand["F-score"])

    plot_moving_avg(train_strat["return"], "Expected Return")
    plot_moving_avg(train_strat["dice"], "Dice Coefficient")
    plot_moving_avg(train_strat["sparsity"], "Sparsity")

    plot_moving_avg_grid(train_strat["return"], train_strat["dice"], train_strat["sparsity"])

    rand_stats = [train_rand["return"], train_rand["dice"], train_rand["sparsity"]]
    strat_stats = [train_strat["return"], train_strat["dice"], train_strat["sparsity"]]
    plot_rand_vs_stratified_grid(rand_stats, strat_stats)
