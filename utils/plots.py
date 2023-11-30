import numpy as np
import matplotlib.pyplot as plt
from dataset.data_prep import load_dict

def barplot_distr(num_dict, title=""):
    """
    :param num_dict: dictionary where keys are the number of fire-present patches
    """
    fire_patch_count = list(num_dict.keys())
    num_images = list(num_dict.values())

    plt.rc('font', size=12)  # controls default text sizes
    plt.rc('axes', titlesize=12)  # fontsize of the axes title
    plt.rc('axes', labelsize=14)  # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
    # plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
    # plt.rc('legend', fontsize=10)
    # plt.rc('figure', titlesize=20)

    # fig = plt.figure(figsize=(10, 5))
    plt.bar(fire_patch_count, num_images, color='blue', width=0.5) # width=0.4
    plt.xlabel("fire patches")
    plt.ylabel("images")
    plt.title(title, fontsize=20)

    plt.show()

def stacked_2barplots(dict1, dict2):
    x = [int(key) for key in dict1.keys()]
    y1 = [value for value in dict1.values()]
    y2 = [value for i, value in enumerate(dict2.values())]

    plt.bar(x, y1, color='blue', label='100-sample dataset')
    plt.bar(x, y2, bottom=y1, color='maroon', label='256-sample dataset')

    plt.xlabel("fire patches")
    plt.ylabel("images")

    plt.yscale("log")

    plt.legend()
    plt.show()

def stacked_3barplots(dict1, dict2, dict3):
    x = [int(key) for key in dict1.keys()]
    y1 = [value for value in dict1.values()]
    y2 = [value for value in dict2.values()]
    y3 = [value for value in dict3.values()]

    y2.extend([0, 0])
    y3.extend([0,0,0])

    plt.bar(x, y1, color='blue', label='6179-sample dataset')
    plt.bar(x, y2, color='maroon', label='1024-sample dataset')
    plt.bar(x, y3, color='darkgreen', label='512-sample dataset')

    plt.xlabel("fire patches")
    plt.ylabel("images")

    plt.yscale("log")

    plt.legend()
    plt.show()

def plot_pixl_hist(distr, scale="linear"):
    """
    Part of EDA.
    distr: list of number of fire pixels per image
    scale: "linear" or "log"
    """
    hist, bins, _ = plt.hist(distr, bins=len(distr))
    plt.title("Distribution of fire pixels")

    if scale == "log":
        plt.close()
        logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
        plt.hist(distr, bins=logbins)
        plt.xscale('log')
        plt.title("Distribution of fire pixels (log scale)")

    plt.xlabel("image count")
    plt.ylabel("fire pixel count")
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
    plt.tight_layout()
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

def plot2curves(stat1, stat2, title):
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

    plt.suptitle(title)
    plt.tight_layout()

    # plt.grid(linestyle=':', which='both')
    plt.show()

def plot_batch_size_curves(stats32, stats64, stats128):
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


if __name__ == "__main__":
    # demo
    dict_100 = load_dict("data/EDA/fp_dict100.pkl")
    barplot_distr(dict_100)
