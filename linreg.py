import numpy as np
from statistics import mean
import matplotlib.pyplot as plt

x = np.array([0, 1, 5, 7, 11, 25, 42])
y = np.array([0, 1, 7, 4, 6, 9, 22])

# y=x line for comparison
u = np.array([0, 42])
v = np.array([0, 42])

numerator = (mean(x) * mean(y)) - mean(x * y)
denominator = ((mean(x)) ** 2) - (mean(x ** 2))

slope = numerator / denominator
b = mean(y) - slope * mean(x)
regression_line = [(slope * xl) + b for xl in x]

print(slope)

plt.plot(u, v)
plt.scatter(x, y)

plt.plot(x, regression_line)
plt.title('Linear Regression Line')
plt.show()
