import numpy as np
import math
import matplotlib.pyplot as plt

R = math.sqrt(math.pi / 2)
a = -2.0

def pdf(x):
    z = x - a
    inside = R * R - z * z
    if inside <= 0:
        return 0.0
    value = math.sqrt(inside)
    norm = 4.0 / (math.pi ** 2)
    return norm * value

def cdf(x):
    z = (x - a) / R
    if z <= -1:
        return 0.0
    if z >= 1:
        return 1.0
    part = z * math.sqrt(1.0 - z * z) + math.asin(z)
    return part / math.pi + 0.5

def sample_rejection(size):
    samples = []
    x_min = a - R
    x_max = a + R
    f_max = (4.0 / (math.pi ** 2)) * R
    while len(samples) < size:
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(0.0, f_max)
        if y <= pdf(x):
            samples.append(x)
    return np.array(samples)

n = 2000
k = 15

data = sample_rejection(n)

sample_mean = float(np.mean(data))
sample_var = float(np.var(data, ddof=1))

counts, bin_edges = np.histogram(data, bins=k, range=(a - R, a + R))
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
counts = counts.astype(float)
rel_freq = counts / n

theo_probs = []
for i in range(len(bin_edges) - 1):
    p = cdf(bin_edges[i + 1]) - cdf(bin_edges[i])
    theo_probs.append(p)
theo_probs = np.array(theo_probs)
expected = theo_probs * n

plt.figure(figsize=(8, 5))
plt.bar(bin_centers, rel_freq, width=(bin_edges[1] - bin_edges[0]), align='center', edgecolor='black')
xs = np.linspace(a - R, a + R, 400)
ys = [pdf(x) for x in xs]
plt.plot(xs, ys, linewidth=2)
plt.xlabel('x')
plt.ylabel('Относительная частота')
plt.title('Гистограмма и теоретическая плотность распределения')
plt.grid(True)
plt.tight_layout()
plt.show()

sorted_data = np.sort(data)
emp_x = sorted_data
emp_y = np.arange(1, n + 1) / n
theo_y = np.array([cdf(x) for x in emp_x])
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
