import matplotlib.pyplot as plt
import numpy as np

a = 1
p = 0.5

l = 10
x = np.linspace(-l, l, 2000)
plt.plot(x,  a * np.abs(x) / (1 + a * np.abs(x)))
plt.plot(x, np.log(1 + a * np.abs(x)))
plt.plot(x,  (np.abs(x) + a) ** p - a**p)  # origin: (|x| + a)^p
plt.plot(x, 1 - np.exp(-np.abs(x) * a))
plt.plot(x, np.ones(x.shape))
plt.show()
