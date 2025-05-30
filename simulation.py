from board import BOARD_LIST, NAME_TO_INDEX
from probabilities import roll_dice, chance_destination_probabilities, community_chest_destination_probabilities

import numpy as np


def single_move(start_position: int, roll: int) -> int:
    """
    Perform a single move on the Monopoly board.
    
    Args:
        start_position (int): Starting position on the board.
        roll (int): The result of the dice roll.
        
    Returns:
        int: New position after the move.
    """
    end_position = (start_position + roll) % len(BOARD_LIST)
    if BOARD_LIST[end_position] == "Go to Jail":
        return NAME_TO_INDEX["Jail"]
    elif BOARD_LIST[end_position] == "Chance":
        dest = np.random.choice(
            list(chance_destination_probabilities.keys()),
            p=list(chance_destination_probabilities.values())
        )
        if dest == "Nearest Station":
            stations = [idx for idx, name in enumerate(BOARD_LIST) if "Station" in name]
            nearest_station = min(
                stations,
                key=lambda idx: (idx - end_position) % len(BOARD_LIST)
            )
            return nearest_station
        elif dest == "Back 3 Spaces":
            return (end_position - 3) % len(BOARD_LIST)
        elif dest == "Stay":
            return end_position
        else:
            return NAME_TO_INDEX[dest]
    elif BOARD_LIST[end_position] == "Community Chest":
        dest = np.random.choice(
            list(community_chest_destination_probabilities.keys()),
            p=list(community_chest_destination_probabilities.values())
        )
        if dest == "Stay":
            return end_position
        else:
            return NAME_TO_INDEX[dest]
    else:
        return end_position

def simulate_moves(num_moves: int, start_position: int = 0) -> np.ndarray:
    """
    Simulate a series of moves on the Monopoly board.
    
    Args:
        num_moves (int): Number of moves to simulate.
        start_position (int): Starting position on the board (default is 0).
        
    Returns:
        np.ndarray: Array of positions after each move.
    """
    positions = np.zeros(num_moves, dtype=int)
    current_position = start_position

    for i in range(num_moves):
        roll = sum(roll_dice([1, 2, 3, 4, 5, 6], num_dice=2))
        
        current_position = single_move(current_position, roll)
        positions[i] = current_position

    return positions

def log_probabilities(num_iterations: int, iteration_size: int, start_position: int = 0) -> np.ndarray:
    """
    Log the probabilities of landing on each square after a number of iterations.
    
    Args:
        num_iterations (int): Number of iterations to simulate.
        iteration_size (int): Number of moves in each iteration.
        start_position (int): Starting position on the board (default is 0).
        
    Returns:
        np.ndarray: Array of probabilities for each square.
    """
    
    position_counts = np.zeros(len(BOARD_LIST), dtype=int)

    for i in range(num_iterations):
        positions = simulate_moves(iteration_size, start_position)
        unique, counts = np.unique(positions, return_counts=True)
        position_counts[unique] += counts
        start_position = positions[-1]
        
        move_count = (i+1) * iteration_size
        probabilities = position_counts / move_count

    print(f"Simulated probabilities after {num_iterations} iterations of {iteration_size} moves:")
    sorted_probs = sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)
    for idx, prob in sorted_probs:
        print(f"{BOARD_LIST[idx]}: {prob:.4f}")
    print()
    
    return probabilities
