import pickle as pkl
import matplotlib.pyplot as plt
import os
from utils import utils
import pickle

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

NUM_SAMPLES = 512
logdir = f"pretrainedResNet/{NUM_SAMPLES}/thres0.05/"
# logdir_strat = f"train_agent/512/stratified/Res/"
# logdir_strat2 = f"train_agent/256/stratified/R_bsz64"
# logdir_strat3 = f"train_agent/256/stratified/R_bsz128"

def load_data(logdir, fname1, fname2):
    with open(os.path.join(logdir, fname1), "rb") as fp:
        train_stats = pkl.load(fp)
    with open(os.path.join(logdir, fname2), "rb") as fp:
        test_stats = pkl.load(fp)
    return train_stats, test_stats

def plot_2stats(stat1, stat2):
    plt.rc('font', size=12)
    plt.rc('axes', labelsize=14)
    plt.subplots(1, 2, figsize=(9, 4))

    plt.subplot(1, 2, 1)
    plt.xlabel('Loss')
    plt.plot(stat1)
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.xlabel('F-score')
    plt.plot(stat2)
    plt.grid(True)

    plt.suptitle(F'training on {NUM_SAMPLES}-sample dataset with fire threshold 5%')
    plt.tight_layout()

    # plt.grid(linestyle=':', which='both')
    plt.show()

def plot_moving_avg(stats, title):

    window = 100
    moving_average = [sum(stats[i:i + window]) / window for i in range(len(stats) - window + 1)]

    plt.plot(stats, label='Original Values')
    plt.plot(moving_average, label=f'Moving Average')
    plt.xlabel('Epochs')
    # plt.ylabel('Metric')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    # plt.tight_layout()
    plt.show()

def plot_moving_avg_grid(reward, dice, sparsity):

    window = 100
    cut = 3000
    moving_average_rw = [sum(reward[i:i + window]) / window for i in range(len(reward) - window + 1)]
    moving_average_dc = [sum(dice[i:i + window]) / window for i in range(len(dice) - window + 1)]
    moving_average_sp = [sum(sparsity[i:i + window]) / window for i in range(len(sparsity) - window + 1)]

    figure, axis = plt.subplots(1, 3, figsize=(14, 4))

    moving_average_rw = moving_average_rw[:cut]
    moving_average_dc = moving_average_dc[:cut]
    moving_average_sp = moving_average_sp[:cut]

    axis[0].plot(reward[:cut], label='Original Values')
    axis[0].plot(moving_average_rw, label=f'Moving Average')
    axis[0].set_xlabel('Epochs')
    axis[0].set_ylabel('Expected Return')
    axis[0].grid(True)

    axis[1].plot(dice[:cut], label='Original Values')
    axis[1].plot(moving_average_dc, label=f'Moving Average')
    # axis[1].set_ylim((0.45, 0.75))
    axis[1].set_xlabel('Epochs')
    axis[1].set_ylabel('Dice Coefficient')
    axis[1].grid(True)

    axis[2].plot(sparsity[:cut], label='Original Values')
    axis[2].plot(moving_average_sp, label=f'Moving Average')
    axis[2].set_xlabel('Epochs')
    axis[2].set_ylabel('Sparsity')
    axis[2].grid(True)

    # plt.suptitle(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_rand_vs_stratified(rand_stats, strat_stats):
    window = 100
    moving_avg_rand = [sum(rand_stats[i:i + window]) / window for i in range(len(rand_stats) - window + 1)]
    moving_avg_strat = [sum(strat_stats[i:i + window]) / window for i in range(len(strat_stats) - window + 1)]

    plt.plot(moving_avg_rand, label='Randomly selected data', color="black")
    plt.plot(moving_avg_strat, label='Stratified data', color="red")
    plt.xlabel('Epochs')
    plt.ylabel('Reward')
    # plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_rand_vs_stratified_grid(rand_stats, strat_stats):
    """
    rand_stats[0]=reward, rand_stats[1]=dice, rand_stats[2]=sparsity
    strat_stats[0]=reward, strat_stats[1]=dice, strat_stats[2]=sparsity
    """
    window = 100
    cut = 2000
    moving_avg_rand_rw = [sum(rand_stats[0][i:i + window]) / window for i in range(len(rand_stats[0]) - window + 1)][:cut]
    moving_avg_rand_dc = [sum(rand_stats[1][i:i + window]) / window for i in range(len(rand_stats[1]) - window + 1)][:cut]
    moving_avg_rand_sp = [sum(rand_stats[2][i:i + window]) / window for i in range(len(rand_stats[2]) - window + 1)][:cut]

    moving_avg_strat_rw = [sum(strat_stats[0][i:i + window]) / window for i in range(len(strat_stats[0]) - window + 1)][:cut]
    moving_avg_strat_dc = [sum(strat_stats[1][i:i + window]) / window for i in range(len(strat_stats[1]) - window + 1)][:cut]
    moving_avg_strat_sp = [sum(strat_stats[2][i:i + window]) / window for i in range(len(strat_stats[2]) - window + 1)][:cut]

    figure, axis = plt.subplots(1, 3, figsize=(14, 4))

    axis[0].plot(moving_avg_rand_rw, label='Randomly selected data', color="black")
    axis[0].plot(moving_avg_strat_rw, label='Stratified data', color="red")
    axis[0].set_xlabel('Epochs')
    axis[0].set_ylabel('Expected Return')
    axis[0].grid(True)

    axis[1].plot(moving_avg_rand_dc, label='Randomly selected data', color="black")
    axis[1].plot(moving_avg_strat_dc, label='Stratified data', color="red")
    # axis[1].set_ylim((0.45, 0.75))
    axis[1].set_xlabel('Epochs')
    axis[1].set_ylabel('Dice Coefficient')
    axis[1].grid(True)

    axis[2].plot(moving_avg_rand_sp[:cut], label='Randomly selected data', color="black")
    axis[2].plot(moving_avg_strat_sp, label='Stratified data', color="red")
    axis[2].set_xlabel('Epochs')
    axis[2].set_ylabel('Sparsity')
    axis[2].grid(True)

    # plt.suptitle(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot3curves(stats32, stats64, stats128):
    """
    stats logged after training of the respective batch size
    """
    window = 100
    cut = 2000
    moving_avg32 = [sum(stats32[i:i + window]) / window for i in range(len(stats32) - window + 1)][:cut]
    moving_avg64 = [sum(stats64[i:i + window]) / window for i in range(len(stats64) - window + 1)][:cut]
    moving_avg128 = [sum(stats128[i:i + window]) / window for i in range(len(stats128) - window + 1)][:cut]

    plt.rc('font', size=12)  # controls default text sizes
    plt.rc('axes', titlesize=12)  # fontsize of the axes title
    plt.rc('axes', labelsize=14)  # fontsize of the x and y labels
    plt.rc('legend', fontsize=14)

    plt.plot(moving_avg32, label='batch size 32', color="dodgerblue")
    plt.plot(moving_avg64, label='batch size 64', color="goldenrod")
    plt.plot(moving_avg128, label='batch size 128', color="mediumvioletred")
    plt.xlabel('Epochs')
    plt.ylabel('Sparsity')
    # plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def concatenate_stats(stats1, stats2):
    """
    stats1 and stats2 are dictionaries of lists
    """
    for key in stats1:
        stats1[key].extend(stats2[key])
    return stats1

if __name__ == "__main__":
    train_stats, test_stats = load_data(logdir,"train_stats_E20", "test_stats_E20")
    # train_strat, test_rand = load_data(logdir_strat, "train512_epoch2000", "test512_epoch2000")

    plot_2stats(train_stats["loss"], train_stats["F-score"])

    # train_strat32, test_strat = load_data(logdir_strat1, "train256_epoch2000", "test256_epoch2000")
    # train_strat64, test_strat = load_data(logdir_strat2, "train256", "test256")
    # train_strat128, test_strat = load_data(logdir_strat3, "train256", "test256")

    # plot_moving_avg(train_strat["return"], "Expected Return")
    # plot_moving_avg(train_strat["dice"], "Dice Coefficient")
    # plot_moving_avg(train_strat["sparsity"], "Sparsity")
    # plot_stat(train_stats["variance"], "Variance")
    # plot_moving_avg_grid(train_strat["return"], train_strat["dice"], train_strat["sparsity"])

    # rand_stats = [train_rand["return"], train_rand["dice"], train_rand["sparsity"]]
    # strat_stats = [train_strat["return"], train_strat["dice"], train_strat["sparsity"]]
    # plot_rand_vs_stratified_grid(rand_stats, strat_stats)

    # plot3curves(train_strat32["sparsity"], train_strat64["sparsity"], train_strat128["sparsity"])