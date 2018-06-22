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

N = len(X)
X = np.array(X)
Y = np.array(Y)

plt.scatter(X, Y)
plt.show()

# 梯度递减
costs = [] # keep track of squared error cost
w = np.random.randn() # randomly initialize w
learning_rate = 0.0000001
# learning_rate = 0.000001
for t in range(1000):
    # update w
    Yhat = X.dot(w)
    delta = Yhat - Y
    w = w - learning_rate*X.T.dot(delta)

    # find and store the cost
    mse = delta.dot(delta) / N
    costs.append(mse)

# plot the costs
plt.plot(costs)
plt.show()

print("final w:", w)

plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()
