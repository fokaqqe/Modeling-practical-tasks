import numpy as np
import math
import matplotlib.pyplot as plt

p_down = 0.5
p_up = 0.1
p_left = 0.2
p_right = 0.2
probs = [p_down, p_up, p_left, p_right]
moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]
h_values = [2, 4, 8, 16, 32]
N = 20000
np.random.seed(0)

def one_walk(h):
    x = 0
    y = h
    steps = 0
    while y > 0:
        r = np.random.rand()
        if r < probs[0]:
            dx, dy = moves[0]
        elif r < probs[0] + probs[1]:
            dx, dy = moves[1]
        elif r < probs[0] + probs[1] + probs[2]:
            dx, dy = moves[2]
        else:
            dx, dy = moves[3]
        x += dx
        y += dy
        steps += 1
    return steps

def empirical_cdf(data):
    s = np.sort(data)
    y = np.arange(1, len(s)+1) / len(s)
    return s, y

def normal_cdf(x, mean, std):
    return 0.5 * (1 + math.erf((x - mean) / (std * math.sqrt(2))))

def ks_statistic(data, cdf_func):
    s = np.sort(data)
    n = len(s)
    ecdf = np.arange(1, n+1) / n
    diffs = [abs(ecdf[i] - cdf_func(s[i])) for i in range(n)]
    return max(diffs)

for h in h_values:
    times = np.empty(N, dtype=int)
    for i in range(N):
        times[i] = one_walk(h)
    mean_t = times.mean()
    std_t = times.std(ddof=0)
    lam_exp = 1 / mean_t
    s, y = empirical_cdf(times)
    ks_norm = ks_statistic(times, lambda x: normal_cdf(x, mean_t, std_t))
    ks_exp = ks_statistic(times, lambda x: 1 - math.exp(-lam_exp * x))
    better = "нормальная" if ks_norm < ks_exp else "экспоненциальная"
    print(f"h = {h}")
    print(f"Среднее τ = {mean_t:.4f}, σ = {std_t:.4f}")
    print(f"KS нормального = {ks_norm:.4f}, KS экспоненциального = {ks_exp:.4f}")
    print(f"Более целесообразная аппроксимация: {better}")
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.hist(times, bins=50, density=True)
    xs = np.linspace(times.min(), times.max(), 200)
    norm_pdf = (1/(std_t*math.sqrt(2*math.pi))) * np.exp(-0.5*((xs-mean_t)/std_t)**2)
    exp_pdf = lam_exp * np.exp(-lam_exp * xs)
    plt.plot(xs, norm_pdf, label='Норм. плотность')
    plt.plot(xs, exp_pdf, label='Экспон. плотность')
    plt.title(f"Гистограмма τ при h={h}")
    plt.legend()
    plt.subplot(1,2,2)
    plt.step(s, y, where='post', label='Эмпир. ФР')
    theor_norm_cdf = [normal_cdf(x, mean_t, std_t) for x in s]
    theor_exp_cdf = [1 - math.exp(-lam_exp * x) for x in s]
    plt.plot(s, theor_norm_cdf, label='Норм. ФР')
    plt.plot(s, theor_exp_cdf, label='Экспон. ФР')
    plt.title(f"Эмпир. и теор. ФР при h={h}")
    plt.legend()
    plt.tight_layout()
    plt.show()
