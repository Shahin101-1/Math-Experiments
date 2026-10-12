import numpy as np
import scipy as sp

# implement Mean value:

number = [[1,2,3,4,5,6],[3,4,5,6,7,8]]
print(np.mean(number))
print(np.mean(number, axis=0))
print(np.mean(number, axis=1))

# implement Variance:

b = [1,3,5,6]
number = [[1,2,3,4,5,6],[3,4,5,6,7,8]]
print(np.var(b))
print(np.var(number))

#Covariance Implementation:

x = np.array([0, 1, 2])
print(np.cov(x))

#correlation Implementation:

vc = [1,2,39,0,8]
vb = [1,2,38,0,8]
np.corrcoef(vc, vb)

#Binomial Distribution Implementation:
from scipy.stats import binom, norm, beta, expon
import matplotlib.pyplot as plt

binom_sim = binom.rvs(n=10, p=0.3, size=1000)
print('data:', binom_sim)
print('mean: %g' % np.mean(binom_sim))
print('SD:  %g' % np.std(binom_sim, ddof=1))

plt.hist(binom_sim, bins=10)
plt.xlabel(('x'))
plt.ylabel('density')
plt.show()

# poisson distribution Implementation:

X = np.random.poisson(lam=2, size=1000)

a = plt.hist(X, bins=15, range=[0, 15])
plt.grid()
plt.show()

# normal distribution Implementation:

mu = 0
sigma = 1
x = np.arange(-5, 5, 0.1)
y = norm.pdf(x, mu, sigma)
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('density')
plt.show()

# THE  END!