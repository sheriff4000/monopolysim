from board import BOARD_LIST, NAME_TO_INDEX
from probabilities import base_roll_probability, transition_matrix
import numpy as np

dice = [1, 2, 3, 4, 5, 6]
num_dice = 2

base_probs = base_roll_probability(dice, num_dice)
transitions = transition_matrix(base_probs, BOARD_LIST)
# transitions = np.hstack((transitions, np.ones((transitions.shape[0], 1))))


# Calculate the steady state distribution
