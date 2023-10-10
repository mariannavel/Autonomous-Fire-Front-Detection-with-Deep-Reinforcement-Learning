import pickle as pkl

with open("train_agent/train_stats_E_3_samples_100", "rb") as fp:
    train_stats = pkl.load(fp)

with open("train_agent/test_stats_E_3_samples_100", "rb") as fp:
    test_stats = pkl.load(fp)

print(test_stats)