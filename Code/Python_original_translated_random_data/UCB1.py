import math
import numpy as np


def UCB1(mu, K, T) -> float:
    pullnumber = np.ones(K, dtype=int)
    averageReward = np.zeros(K, dtype=float)

    for t in range(0, K):
        averageReward[t] = mu[t] + np.random.randn()

    for t in range(K, T):
        UCB = averageReward +\
              np.sqrt(2 * np.divide(math.log(T), pullnumber))
        pos = UCB.argmax(0)
        weight = 1 / (pullnumber[pos] + 1)
        averageReward[pos] = (1 - weight) * averageReward[pos] + \
                             weight * (mu[pos] + np.random.randn())
        pullnumber[pos] = pullnumber[pos] + 1

    regret = (mu[0] - mu[1:]).dot(pullnumber[1:])
    return regret
