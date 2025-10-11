import numpy as np
import math
import matplotlib.pyplot as plt

mu = 3.0
sigma = math.sqrt(0.1)
n = 2000
k = 15

def normal_central_limit(size):
    samples = []
    for _ in range(size):
        u = np.random.rand(12)
        z = np.sum(u) - 6.0
        x = mu + sigma * z
        samples.append(x)
    return np.array(samples)

def normal_pdf(x):
    return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def normal_cdf(x):
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))

data = normal_central_limit(n)

sample_mean = float(np.mean(data))
sample_var = float(np.var(data, ddof=1))

counts, bin_edges = np.histogram(data, bins=k)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
counts = counts.astype(float)
rel_freq = counts / n

theo_probs = []
for i in range(len(bin_edges) - 1):
    p = normal_cdf(bin_edges[i + 1]) - normal_cdf(bin_edges[i])
    theo_probs.append(p)
theo_probs = np.array(theo_probs)
expected = theo_probs * n

plt.figure(figsize=(8, 5))
plt.bar(bin_centers, rel_freq, width=(bin_edges[1] - bin_edges[0]), align='center', edgecolor='black')
xs = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)
ys = [normal_pdf(x) for x in xs]
plt.plot(xs, ys, linewidth=2)
plt.xlabel('x')
plt.ylabel('Относительная частота')
plt.title('Гистограмма и теоретическая плотность нормального распределения')
plt.grid(True)
plt.tight_layout()
plt.show()

sorted_data = np.sort(data)
emp_x = sorted_data
emp_y = np.arange(1, n + 1) / n
theo_y = np.array([normal_cdf(x) for x in emp_x])
D_plus = np.max(emp_y - theo_y)
D_minus = np.max(theo_y - (np.arange(0, n) / n))
D = max(D_plus, D_minus)
D_crit = 1.36 / math.sqrt(n)

chi2 = 0.0
for obs, exp in zip(counts, expected):
    if exp > 0:
        chi2 += (obs - exp) ** 2 / exp

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
