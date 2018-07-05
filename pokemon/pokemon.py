from __future__ import print_function, division
from builtins import range
import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []
# 准备数据
for line in open('pokemon.csv'):
    r = line.split(',')
    # if r[2] == 'cp':
    #     continue
    # if r[1] == 'Pidgey':
    # if r[1] == 'Weedle':
    if r[1] == 'Caterpie':
    # if r[1] == 'Eevee':
        X.append(int(r[2]))
        Y.append(int(r[14]))

X = np.array(X)
Y = np.array(Y)

plt.scatter(X, Y)
plt.show()

# 最小二乘法直接求解
denominator = X.dot(X) - X.mean() * X.sum()
w = ( X.dot(Y) - Y.mean()*X.sum() ) / denominator
b = ( Y.mean() * X.dot(X) - X.mean() * X.dot(Y) ) / denominator

Yhat = w * X + b

print("w: ", w, ",b", b)

plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()

d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("The r-squared is:", r2)


