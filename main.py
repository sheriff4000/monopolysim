from board import BOARD_LIST, NAME_TO_INDEX
from probabilities import base_roll_probability, transition_matrix
import numpy as np

dice = [1, 2, 3, 4, 5, 6]
num_dice = 2

base_probs = base_roll_probability(dice, num_dice)
transitions = transition_matrix(base_probs, BOARD_LIST)
print(transitions.shape)

# we want to solve the equation πP = π
# where π (row vector) is the steady state distribution and P is the transition matrix 
# we can rewrite this as π(P - I) = 0
# and solve for π by introducing a new matrix Q = (P - I) with an extra column of ones appended
# and a row vector of zeros c on the right-hand side with the last element being 1
# This results in π = cQ^-1

Q = transitions - np.eye(transitions.shape[0])

# Add a column of ones to force a single solution where the sum of the probabilities is 1
Q = np.hstack((Q, np.ones((Q.shape[0], 1))))
print(Q.shape)

# Making the rhs a row vector of zeros aside from the last element - to enforce the sum of the probabilities to be 1
c = np.zeros((1, Q.shape[1]))
c[0, -1] = 1
print(c.shape)

Q_inverse = np.linalg.pinv(Q)
print(Q_inverse.shape)

# Calculate the steady state distribution as cQ^-1
steady_state = c @ Q_inverse

# Normalize the steady state distribution
steady_state /= np.sum(steady_state[0, :]) 
print(steady_state)

print(sum(steady_state[0, :]))  # Should be 1

for i, prob in enumerate(steady_state[0, :]):
    print(f"{BOARD_LIST[i]}: {prob:.4f}")
