import numpy as np
from itertools import product
from board import BOARD_LIST, NAME_TO_INDEX
from typing import Dict

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

def transition_probabilities(base_probabilities: Dict[int, float], board_list: list[str]):
    """
    Calculate the transition probabilities based on the base roll probabilities.
    :param base_probabilities: Base roll probabilities
    :param board_list: List of board spaces
    :return: Transition probability matrix
    """
    transition_matrix = np.zeros((len(board_list), len(board_list)))

    for start_idx in range(len(board_list)):
        for roll_sum, prob in base_probabilities.items():
            end_idx = (start_idx + roll_sum) % len(board_list)
            # Check for special cases
            if board_list[end_idx] == "Go to Jail":
                transition_matrix[start_idx][NAME_TO_INDEX["Jail"]] += prob
            elif board_list[end_idx] == "Chance":
                # There are 16 Chance cards, 1/16 chance to draw each
                chance_destinations = ["Go", "Mayfair", "Jail", "Trafalgar Square", "Pall Mall", "King's Cross Station"]
                for destination in chance_destinations:
                    transition_matrix[start_idx][NAME_TO_INDEX[destination]] += (1/16) * prob
                # nearest station card occurs twice in the deck, so probablity is 2/16 = 1/8
                nearest_station = min(
                    (idx for idx, space in enumerate(board_list) if "Station" in space),
                    key=lambda idx: (idx - end_idx) % len(board_list) if (idx - end_idx) > 0 else float('inf')
                )
                transition_matrix[start_idx][nearest_station] += (2/16) * prob

                back_three = end_idx - 3
                if back_three < 0:
                    back_three += len(board_list)
                # Don't need to check if back three is a chance as this is not possible
                if board_list[back_three] == "Community Chest":
                    # 1/16 chance to draw each Community Chest card
                    community_destinations = ["Go", "Jail"]
                    for destination in community_destinations:
                        #1/16 to draw "back three" multiplied by 1/16 to get an advancing community chest card
                        transition_matrix[start_idx][NAME_TO_INDEX[destination]] += (1/16) * (1/16) * prob

                # 7/16 chance to draw a card that doesn't move you
                transition_matrix[start_idx][end_idx] += (7/16) * prob
                
            elif board_list[end_idx] == "Community Chest":
                # There are 16 Community Chest cards, 1/16 chance to draw each
                community_destinations = ["Go", "Jail"]
                for destination in community_destinations:
                    #1/16 to draw "back three" multiplied by 1/16 to get an advancing community chest card
                    transition_matrix[start_idx][NAME_TO_INDEX[destination]] += (1/16) * (1/16) * prob

                # 14/16 chance to draw a card that doesn't move you
                transition_matrix[start_idx][end_idx] += (14/16) * prob
            else:
                # Normal case, just add the probability to the transition matrix
                transition_matrix[start_idx][end_idx] += prob

    return transition_matrix



def roll_dice(dice_sides, num_dice):
    """
    Simulate rolling a number of dice.
    :param num_dice: Number of dice to roll
    :return: Array of rolled values
    """
    return np.random.choice(dice_sides, size=num_dice, replace=True)
