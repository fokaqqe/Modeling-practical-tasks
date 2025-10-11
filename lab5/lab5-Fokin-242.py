import numpy as np
import math
import matplotlib.pyplot as plt

n = 2000
k = 15

lam_weibull = 1.5
k_weibull = 2.0

alpha_gamma = 3.0
beta_gamma = 2.0

def weibull_generate(size):
    u = np.random.rand(size)
    return lam_weibull * (-np.log(1 - u)) ** (1 / k_weibull)

def weibull_pdf(x):
    if x < 0:
        return 0.0
    return (k_weibull / lam_weibull) * (x / lam_weibull) ** (k_weibull - 1) * math.exp(- (x / lam_weibull) ** k_weibull)

def weibull_cdf(x):
    if x < 0:
        return 0.0
    return 1 - math.exp(- (x / lam_weibull) ** k_weibull)

# --- Гамма ---
def gamma_generate(size):
    data = []
    for _ in range(size):
        prod = 1.0
        for _ in range(int(alpha_gamma)):
            prod *= np.random.rand()
        x = -math.log(prod) * beta_gamma
        data.append(x)
    return np.array(data)

def gamma_pdf(x):
    if x < 0:
        return 0.0
    return (x ** (alpha_gamma - 1) * math.exp(-x / beta_gamma)) / (math.gamma(alpha_gamma) * (beta_gamma ** alpha_gamma))

def gamma_cdf(x):
    if x < 0:
        return 0.0
    total = 0
    for i in range(int(alpha_gamma)):
        total += ((x / beta_gamma) ** i) / math.factorial(i)
    return 1 - math.exp(-x / beta_gamma) * total

def analyze(data, pdf_func, cdf_func, title):
    sample_mean = float(np.mean(data))
    sample_var = float(np.var(data, ddof=1))

    counts, bin_edges = np.histogram(data, bins=k)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    counts = counts.astype(float)
    rel_freq = counts / n

    theo_probs = []
    for i in range(len(bin_edges) - 1):
        p = cdf_func(bin_edges[i + 1]) - cdf_func(bin_edges[i])
        theo_probs.append(p)
    theo_probs = np.array(theo_probs)
    expected = theo_probs * n

    plt.figure(figsize=(8, 5))
    plt.bar(bin_centers, rel_freq, width=(bin_edges[1] - bin_edges[0]), align='center', edgecolor='black')
    xs = np.linspace(min(data), max(data), 400)
    ys = [pdf_func(x) for x in xs]
    plt.plot(xs, ys, linewidth=2)
    plt.xlabel('x')
    plt.ylabel('Относительная частота')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    sorted_data = np.sort(data)
    emp_y = np.arange(1, n + 1) / n
    theo_y = np.array([cdf_func(x) for x in sorted_data])
    D_plus = np.max(emp_y - theo_y)
    D_minus = np.max(theo_y - (np.arange(0, n) / n))
    D = max(D_plus, D_minus)
    D_crit = 1.36 / math.sqrt(n)

    chi2 = 0.0
    for obs, exp in zip(counts, expected):
        if exp > 0:
            chi2 += (obs - exp) ** 2 / exp

    print(title)
    print('Объем выборки:', n)
    print('Выборочное среднее:', sample_mean)
    print('Выборочная дисперсия:', sample_var)
    print('Критерий Колмогорова-Смирнова D:', D)
    print('Критическое значение D (α=0.05):', D_crit)
    if D > D_crit:
        print('Гипотеза H0 отвергается при уровне значимости 0.05')
    else:
        print('Гипотеза H0 не отвергается при уровне значимости 0.05')
    print('Статистика χ²:', chi2)
    print('Количество интервалов:', k)
    print('Ожидаемые частоты по интервалам:', np.round(expected, 3))
    print()

data_weibull = weibull_generate(n)
analyze(data_weibull, weibull_pdf, weibull_cdf, 'Распределение Вейбулла')

data_gamma = gamma_generate(n)
analyze(data_gamma, gamma_pdf, gamma_cdf, 'Гамма-распределение')
