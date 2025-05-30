import matplotlib.pyplot as plt

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
