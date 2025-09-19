import random
import matplotlib.pyplot as plt
import math

N = 1000
m = 10
Y0 = 7
a, c, M = 1229, 1, 2048

# лкг
Y = [Y0]
for i in range(1, N):
    next_val = (a * Y[i-1] + c) % M
    Y.append(next_val)
X = [y / M for y in Y]


# 1. Критерий Пирсона
counts = [0] * m
for x in X:
    index = int(x * m)
    if index == m:
        index = m - 1
    counts[index] += 1

expected = N / m
chi2 = sum((obs - expected) ** 2 / expected for obs in counts)
print("X^2 (Пирсона):", chi2)

# 2. Критерий Калмогорова
X_sorted = sorted(X)
D = max(abs((i+1)/N - X_sorted[i]) for i in range(N))
print("D (Калмогоров):", D)

# 3. Тест длины серий нулей
p = 0.65
binary = [0 if x <= p else 1 for x in X]

# серии нулей
series_lengths = []
count = 0
for bit in binary:
    if bit == 0:
        count += 1
    else:
        if count > 0:
            series_lengths.append(count)
        count = 0
if count > 0:
    series_lengths.append(count)

from collections import Counter
series_count = Counter(series_lengths)

print("Распределение длин серий нулей:")
for length, freq in sorted(series_count.items()):
    print(f"Длина {length}: {freq}")

print("\nТеоретическая вероятность серий:")
for k in range(1, 6):
    print(f"k={k}, P={((1-p)**k) * p:.4f}")
