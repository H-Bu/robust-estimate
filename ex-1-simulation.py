import math
from scipy.special import betaincinv
import matplotlib.pyplot as plt
import numpy as np


def okamoto(eps, delta):
    return math.ceil(math.log(2 / delta) / (2 * eps * eps))


def strong(p, eps):
    # f(p,eps)
    assert eps < 1 - p
    return (p / (p + eps)) ** (p + eps) * ((1 - p) / (1 - p - eps)) ** (1 - p - eps)


def binary(f1, f2, delta, low, high):
    # find the least n, s.t. f1^n+f2^n<=delta
    assert f1**high + f2**high <= delta
    while high - low > 1:
        mid = (high + low) // 2
        if f1 ** mid + f2 ** mid > delta:
            low = mid
        else:
            high = mid
    return high


def bound_estimate(cp_l, cp_h, delta, eps):
    if cp_h <= eps:
        f = strong(cp_h, eps)
        return math.ceil(math.log(delta, f))
    elif cp_l >= 1 - eps:
        f = strong(1 - cp_l, eps)
        return math.ceil(math.log(delta, f))
    elif (1 - eps) / 2 <= cp_h <= (1 + eps) / 2 or (1 - eps) / 2 <= cp_l <= (1 + eps) / 2 or (
            cp_h >= (1 + eps) / 2 and cp_l <= (1 - eps) / 2):
        return okamoto(eps, delta)
    else:
        if eps < cp_h < (1 - eps) / 2:
            pp = cp_h
        else:
            assert (1 + eps) / 2 < cp_l < 1 - eps
            pp = cp_l
        f1 = strong(pp, eps)
        f2 = strong(1 - pp, eps)
        return binary(f1, f2, delta, 0, okamoto(eps, delta))


def cp_int(N, Np, delta):
    # Clopper-Pearson confidence interval
    if Np == 0:
        cp_l = 0
    else:
        cp_l = betaincinv(Np, N - Np + 1, delta / 2)
    if Np == N:
        cp_h = 1
    else:
        cp_h = betaincinv(Np + 1, N - Np, 1 - delta / 2)
    return cp_l, cp_h


eps = 0.05
delta = 0.05
delta1 = 0.05 * delta
assert 0 < eps < 1 / 3

M = okamoto(eps, delta)

p_list = np.arange(0.0, 1.002, 0.002)

p_xx = np.zeros_like(p_list)
p_oo = np.ones_like(p_list) * M
round_num = 100

for ii in range(len(p_list)):
    p = p_list[ii]
    count = 0
    for kk in range(round_num):
        # first stage
        size_1 = max(min(math.ceil(0.01*M), 100), 10)
        count += size_1
        count_1 = np.random.binomial(size_1, p, 1)[0]
        p_1 = count_1 / size_1

        # second stage
        N_list = [round((i + 1) * M / 100) for i in range(20)]  # 20 candidates
        length = len(N_list)
        N_num = N_list.copy()  # total_cost
        for i in range(length):
            N = N_list[i]
            Np = round(N * p_1)
            cp_l, cp_h = cp_int(N, Np, delta1)
            N_num[i] += bound_estimate(cp_l, cp_h, (delta - delta1) / (1 - delta1), eps)

        m = min(N_num)
        flag = False
        if m > M:  # directly use the Okamoto bound
            count += M
        else:
            flag = True

        if flag:
            N = N_list[N_num.index(m)]
            count += N
            Np = np.random.binomial(N, p, 1)[0]
            cp_l, cp_h = cp_int(N, Np, delta1)
            count += bound_estimate(cp_l, cp_h, (delta - delta1) / (1 - delta1), eps)
    p_xx[ii] = count / round_num

print(sum(p_xx)/sum(p_oo))
print(max(p_xx)/max(p_oo))
print(min(p_xx)/min(p_oo))
# print(sum(p_xx))
plt.plot(p_list, p_xx)
plt.plot(p_list, p_oo)
plt.xlabel('p')
plt.ylabel('sampling number')
plt.show()
