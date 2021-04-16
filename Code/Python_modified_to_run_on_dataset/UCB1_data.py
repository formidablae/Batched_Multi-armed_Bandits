import math
import numpy as np


def UCB1_data(ratingsOfChosenBanditWithGenres, mu, K, T) -> float:
    pullnumber = np.ones(K, dtype=int)
    averageReward = np.zeros(K, dtype=float)

    for k in range(0, K):
        rewardsOfFirstTimeHorizon = ratingsOfChosenBanditWithGenres[
            (ratingsOfChosenBanditWithGenres['Arm_K'] == k + 1) &
            (ratingsOfChosenBanditWithGenres['Time_T'] == 1)
            ]['Reward_mu'].to_numpy()
        averageReward[k] = mu[k] + rewardsOfFirstTimeHorizon

    for t in range(1, T):
        UCB = averageReward +\
              np.sqrt(2 * np.divide(math.log(T), pullnumber))
        pos = UCB.argmax(0)
        weight = 1 / (pullnumber[pos] + 1)
        rewardsOfThisTimeHorizon = ratingsOfChosenBanditWithGenres[
            (ratingsOfChosenBanditWithGenres['Arm_K'] == pos + 1) &
            (ratingsOfChosenBanditWithGenres['Time_T'] == t)
            ]['Reward_mu'].to_numpy()
        averageReward[pos] = (1 - weight) * averageReward[pos] + \
                             weight * (mu[pos] + rewardsOfThisTimeHorizon)

        pullnumber[pos] = pullnumber[pos] + 1

    regret = (mu[0] - mu[1:]).dot(pullnumber[1:])
    return regret
