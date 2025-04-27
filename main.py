from board import BOARD_LIST, NAME_TO_INDEX
from probabilities import base_roll_probability, transition_matrix
import numpy as np
import matplotlib.pyplot as plt


dice = [1, 2, 3, 4, 5, 6]
num_dice = 2

base_probs = base_roll_probability(dice, num_dice)
transitions = transition_matrix(base_probs, BOARD_LIST)
print("Transition Matrix Shape: ", transitions.shape)

def steady_state_distribution(transitions):
    """
    Compute the steady-state distribution π of a Markov chain with transition matrix P.

    We solve for π such that:
        πP = π  and  sum(π) = 1

    This is equivalent to solving:
        π(P - I) = 0
    with an additional constraint:
        sum(π) = 1

    To enforce the constraint:
    - Replace the last column of (P - I) with ones
    - Set the corresponding entry in the right-hand side vector to 1

    Args:
        P (np.ndarray): (n, n) row-stochastic transition matrix

    Returns:
        np.ndarray: (1, n) steady-state distribution
    """
    n = transitions.shape[0]
    # Set up the matrix for π(P - I) = 0
    A = transitions - np.eye(n)
    
    # Replace one column to enforce the sum(π) = 1 constraint
    A[:, -1] = np.ones(n)
    
    # Right-hand side vector
    b = np.zeros(n)
    b[-1] = 1
    
    # Solve the system (taking A^T linalg.solve solve Ax = b NOT xA = b)
    steady_state = np.linalg.solve(A.T, b)  # Solve for π^T, transpose A
    
    return steady_state


def plot_monopoly_heatmap(steady_state, board_list):
    """
    Plot a heatmap of the steady state distribution on the Monopoly board.
    """
    n = len(steady_state)
    
    # Try to reshape into something roughly square
    width = 8
    height = 5  # Because 8x5 = 40 squares

    data = steady_state.reshape((height, width))
    
    plt.figure(figsize=(12, 6))
    plt.imshow(data, cmap='plasma', interpolation='nearest')
    plt.colorbar(label='Probability')
    
    # Add text labels
    for i in range(height):
        for j in range(width):
            idx = i * width + j
            label = board_list[idx][:6]  # Shorten names to fit
            plt.text(j, i, f"{label}\n{steady_state[idx]:.3f}",
                     ha='center', va='center', color='white', fontsize=8)
    
    plt.title('Monopoly Steady State Distribution Heatmap')
    plt.axis('off')
    plt.show()

steady_state = steady_state_distribution(transitions)
print(steady_state.shape)  # Should be (1, 40)
print(steady_state[:])  # Should be a 1D array of probabilities

print(sum(steady_state[:]))  # Should be 1

sorted_indices = np.argsort(-steady_state[:])  # descending order
for i in sorted_indices:
    print(f"{BOARD_LIST[i]}: {steady_state[i]:.4f}")
    
plot_monopoly_heatmap(steady_state, BOARD_LIST)
