import random
import matplotlib.pyplot as plt


N = 1000      # объем выборки
m = 10        # число интервалов
Y0 = 7        # начальное значение
a, c, M = 1229, 1, 2048   # параметры ЛКГ

# лкг
Y = [Y0]
for i in range(1, N):
    next_val = (a * Y[i-1] + c) % M
    Y.append(next_val)
X = [y / M for y in Y]
print("Первые 20 сгенерированных чисел:")
print(X[:20])


counts = [0] * m
for x in X:
    index = int(x * m)  
    if index == m:      
        index = m - 1
    counts[index] += 1

freqs = [cnt / N for cnt in counts]

cdf = []
s = 0
for f in freqs:
    s += f
    cdf.append(s)


mean = sum(X) / N
moment2 = sum(x**2 for x in X) / N
moment3 = sum(x**3 for x in X) / N
variance = sum((x - mean)**2 for x in X) / N

print("Мат. ожидание:", mean)
print("Дисперсия:", variance)
print("Второй момент:", moment2)
print("Третий момент:", moment3)
print("\nТеоретические значения:")
print("M = 0.5,  D = 0.0833,  μ2 = 0.333,  μ3 = 0.25")


# Гистограмма
plt.bar(range(m), freqs, width=0.9, edgecolor="black")
plt.title("Гистограмма")
plt.xlabel("Интервалы")
plt.ylabel("Частота")
plt.show()

# статистическая функция распределения
plt.step(range(m), cdf, where="mid")
plt.title("статистическая функция распределения")
plt.xlabel("Интервалы")
plt.ylabel("F*(x)")
plt.show()
