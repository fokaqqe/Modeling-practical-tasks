import numpy as np
import math

r = 0.1
p = 0.8
n = 10
N = 100000
beta = 0.95

results_A = 0
results_B = 0
results_C = 0

for _ in range(N):
    defects = np.random.rand(n) < r
    detected = defects & (np.random.rand(n) < p)
    k = np.sum(detected)
    if k == 0:
        results_A += 1
    if k == 2:
        results_B += 1
    if k >= 2:
        results_C += 1

prob_A = results_A / N
prob_B = results_B / N
prob_C = results_C / N

def conf_interval(prob):
    z = 1.96
    d = z * math.sqrt(prob * (1 - prob) / N)
    return prob - d, prob + d

a_low, a_high = conf_interval(prob_A)
b_low, b_high = conf_interval(prob_B)
c_low, c_high = conf_interval(prob_C)

print("Оценка вероятности события A:", prob_A)
print("Доверительный интервал:", a_low, a_high)
print("Оценка вероятности события B:", prob_B)
print("Доверительный интервал:", b_low, b_high)
print("Оценка вероятности события C:", prob_C)
print("Доверительный интервал:", c_low, c_high)

p_detect = r * p
true_A = (1 - p_detect) ** n
true_B = math.comb(n, 2) * (p_detect ** 2) * ((1 - p_detect) ** (n - 2))
true_C = 1 - ((1 - p_detect) ** n) - n * p_detect * ((1 - p_detect) ** (n - 1))

print("Аналитическое значение A:", true_A)
print("Аналитическое значение B:", true_B)
print("Аналитическое значение C:", true_C)
