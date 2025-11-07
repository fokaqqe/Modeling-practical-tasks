import numpy as np
import math
import matplotlib.pyplot as plt

p_down = 0.5
p_up = 0.1
p_left = 0.2
p_right = 0.2
N = 10000
beta = 0.95
z = 1.96
h_values = [2, 4, 10, 20, 40, 80]
np.random.seed(0)

def one_walk(h):
    x = 0
    y = h
    steps = 0
    while y > 0:
        r = np.random.rand()
        if r < p_down:
            y -= 1
        elif r < p_down + p_up:
            y += 1
        elif r < p_down + p_up + p_left:
            x -= 1
        else:
            x += 1
        steps += 1
    return steps

def conf_interval_mean(data):
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    d = z * std / math.sqrt(len(data))
    return mean, std, mean - d, mean + d

def conf_interval_var(data):
    var = np.var(data, ddof=1)
    n = len(data)
    chi2_low = (n - 1) * var / (0.5 * (1 + beta))
    chi2_high = (n - 1) * var / (0.5 * (1 - beta))
    return var, chi2_low, chi2_high

def plan_experiment_mean(h, eps_rel):
    sample = np.array([one_walk(h) for _ in range(1000)])
    mean, std, _, _ = conf_interval_mean(sample)
    n = math.ceil((z * std / (eps_rel * mean)) ** 2)
    return n

def plan_experiment_var(h, eps_rel):
    sample = np.array([one_walk(h) for _ in range(1000)])
    var = np.var(sample, ddof=1)
    n = math.ceil(2 * (z / eps_rel) ** 2)
    return n

for h in h_values:
    n_mean = plan_experiment_mean(h, 0.05)
    n_var = plan_experiment_var(h, 0.1)
    data_mean = np.array([one_walk(h) for _ in range(n_mean)])
    data_var = np.array([one_walk(h) for _ in range(n_var)])
    mean, std, low_m, high_m = conf_interval_mean(data_mean)
    var, low_v, high_v = conf_interval_var(data_var)
    print(f"h = {h}")
    print(f"Планируемое число опытов для оценки среднего: {n_mean}")
    print(f"Среднее τ = {mean:.2f}, доверительный интервал: [{low_m:.2f}, {high_m:.2f}]")
    print(f"Планируемое число опытов для оценки дисперсии: {n_var}")
    print(f"Дисперсия τ = {var:.2f}")
    print()
    plt.hist(data_mean, bins=40, density=True)
    plt.title(f"Гистограмма времени падения τ при h={h}")
    plt.xlabel("τ")
    plt.ylabel("Плотность")
    plt.show()
