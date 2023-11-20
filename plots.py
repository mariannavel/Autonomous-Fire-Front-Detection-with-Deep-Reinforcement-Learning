import numpy as np
import matplotlib.pyplot as plt
from data_prep import load_dict

def barplot(num_dict, title=""):
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


if __name__ == "__main__":
    # demo
    dict_100 = load_dict("data/EDA/fp_dict100.pkl")
    barplot(dict_100)
