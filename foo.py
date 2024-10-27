import random
import math

random.seed(0)

def normalize_ratios(ratios):
    total = sum(ratios)
    if total > 0:
        return [r / total for r in ratios]
    else:
        return [0] * len(ratios)

def weighted_choice(weights):
    total = sum(weights)
    rand = random.uniform(0, total)

    cum = 0
    for index, value in enumerate(weights):
        cum += value
        if rand < cum:
            return index

    return len(weights) - 1

def expo(rate):
    U = random.random()
    return -math.log(1 - U) / rate

def gen_to_trans(s):
    n = len(s)
    m = []
    for _ in range(n):
        x = []
        for _ in range(n + 1):
            x.append(0)
        m.append(x)

    for i in range(n):
        row_sum = sum(s[i])
        if row_sum < 0:
            m[i][n] = row_sum / s[i][i]  # Probability of transitioning to the absorbing state
            for j in range(n):
                if i != j:
                    m[i][j] = -s[i][j] / s[i][i]  # Probability of transitioning to state j

    return m

def phase_type(alpha, s):
    trans = gen_to_trans(s)
    print(trans)
    state = weighted_choice(normalize_ratios(alpha))
    ret = 0

    while state < len(s):
        ret += expo(-s[state][state])  # Time spent in the current state
        state = weighted_choice(normalize_ratios(trans[state]))

    return ret

# Example usage

# Initial probability distribution for starting in each state
alpha = [0.1,0.9,0]

# Subgenerator matrix `s`
s = [
    [-6,  2,  0],
    [ 0, -9,  1],
    [ 0,  0, -3],
]

print(phase_type(alpha, s))
exit()

import matplotlib.pyplot as plt

# Sample output from your custom RNG (replace this with actual values)
random_numbers = [phase_type(alpha, s) for _ in range(1000)]  # Replace 'your_rng_output' with your RNG's actual output

# Plot the distribution
plt.figure(figsize=(8, 5))
plt.hist(random_numbers, bins=30, edgecolor='k', alpha=0.7)
plt.title('Histogram of Custom RNG Output')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

