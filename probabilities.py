import numpy as np
from itertools import product
from board import BOARD_LIST, NAME_TO_INDEX
from typing import Dict, List

dice = [1, 2, 3, 4, 5, 6]
num_dice = 2
def base_roll_probability(dice, num_dice):
    """
    Calculate the probability of rolling a sum with a given number of dice.
    :param dice: List of possible dice values
    :param num_dice: Number of dice to roll
    :return: Dictionary with sums as keys and their probabilities as values
    """
    probability_map = {}
    # Generate all possible combinations of dice rolls
    all_rolls = product(dice, repeat=num_dice)

    for roll in all_rolls:
        # Calculate the sum of the rolled values
        roll_sum = sum(roll)
        probability_map[roll_sum] = probability_map.get(roll_sum, 0) + 1
        
    total_rolls = sum(probability_map.values()) 

    for roll in probability_map:
        probability_map[roll] /= total_rolls

    return probability_map

def handle_community_chest(end_idx: int, prob: float, matrix: np.ndarray, start_idx: int):
    # 2 cards move you (Go, Jail), rest don’t
    move_cards = ["Go", "Jail"]
    for dest in move_cards:
        matrix[start_idx][NAME_TO_INDEX[dest]] += (1/16) * prob
    matrix[start_idx][end_idx] += (14/16) * prob

def handle_chance(end_idx: int, prob: float, matrix: np.ndarray, board_list: List[str], start_idx: int):
    # Direct-move cards
    chance_destinations = {
        "Go": 1, "Mayfair": 1, "Jail": 1, 
        "Trafalgar Square": 1, "Pall Mall": 1, "King's Cross Station": 1
    }

    for dest, count in chance_destinations.items():
        matrix[start_idx][NAME_TO_INDEX[dest]] += (count / 16) * prob

    # Nearest station (2 cards)
    stations = [idx for idx, name in enumerate(board_list) if "Station" in name]
    nearest_station = min(
        stations,
        key=lambda idx: (idx - end_idx) % len(board_list)
    )
    matrix[start_idx][nearest_station] += (2 / 16) * prob

    # Go back 3 spaces (1 card)
    back_three = (end_idx - 3) % len(board_list)
    if board_list[back_three] == "Community Chest":
        # Nested chest draw
        matrix[start_idx][NAME_TO_INDEX["Go"]] += (1/16) * (1/16) * prob
        matrix[start_idx][NAME_TO_INDEX["Jail"]] += (1/16) * (1/16) * prob
        matrix[start_idx][back_three] += (1/16) * (14/16) * prob
    else:
        matrix[start_idx][back_three] += (1/16) * prob

    # Remaining 7/16: stay on the square
    matrix[start_idx][end_idx] += (7 / 16) * prob

def transition_matrix(base_probabilities: Dict[int, float], board_list: list[str]):
    """
    Calculate the transition probabilities based on the base roll probabilities.
    :param base_probabilities: Base roll probabilities
    :param board_list: List of board spaces
    :return: Transition probability matrix
    """
    n = len(board_list)
    matrix = np.zeros((n, n))
    #TODO expand this to consider double rolls
    for start_idx in range(n):
        for roll_sum, prob in base_probabilities.items():
            end_idx = (start_idx + roll_sum) % n
            square = board_list[end_idx]

            if square == "Go to Jail":
                matrix[start_idx][NAME_TO_INDEX["Jail"]] += prob
            elif square == "Chance":
                handle_chance(end_idx, prob, matrix, board_list, start_idx)
            elif square == "Community Chest":
                handle_community_chest(end_idx, prob, matrix, start_idx)
            else:
                matrix[start_idx][end_idx] += prob

    # Validation: Check all rows sum to 1
    for i in range(n):
        row_sum = np.sum(matrix[i])
        if not np.isclose(row_sum, 1.0):
            print(f"Warning: Row {i} ('{board_list[i]}') sums to {row_sum:.6f}")
            # Optional: Normalize it
            matrix[i] /= row_sum

    return matrix

def roll_dice(dice_sides, num_dice):
    """
    Simulate rolling a number of dice.
    :param num_dice: Number of dice to roll
    :return: Array of rolled values
    """
    return np.random.choice(dice_sides, size=num_dice, replace=True)
