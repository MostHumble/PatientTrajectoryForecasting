import matplotlib.pyplot as plt
import statistics
from typing import Union, Tuple, List
import numpy as np

def stats(Pairs : List[Tuple[List[List[int]]]], is_torch: bool = False) -> Union[Tuple[List[int], List[int]], Tuple[int, int]] :
    """
    Calculates statistics based on the length of pairs in the given list.

    Args:
        Pairs (list): A list of pairs.
        max_length (int, optional): The maximum length threshold. Defaults to 600.
        is_torch (bool, optional): Indicates whether the pairs are in torch format. Defaults to False.

    Returns:
        bool: True if any pair in the list has a length greater than max_length, False otherwise.
    """
    x, y = [], []
    for (source_sequence,  target_sequence) in Pairs:
        x.append(len(source_sequence))
        y.append(len(target_sequence))


    # Calculating statistics
    mean_source = statistics.mean(x)
    mean_target = statistics.mean(y)
    std_source = statistics.stdev(x)
    std_target = statistics.stdev(y)

    print("\nStatistics of the input and output data:")
    print(f"\nMean sequence length of source sequences : {mean_source:.2f} ± {std_source:.2f}")
    print(f"Mean sequence length of target sequences : {mean_target:.2f} ± {std_target:.2f}")

    # Plotting distribution of lengths
    plt.figure(figsize = (10, 5))
    plt.hist([x, y], bins = 30, color=['blue', 'green'], label = ['source sequences', 'target sequences'])
    plt.title('Distribution of Sequence Lengths')
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()
    return x,y
        
    
    
def token_count_distributions(source_lenghts : List[int], target_lenghts : List[int] , bins : int = 100, xlim : int = 50) -> None :
    """
    Plots the distribution of lenghts.

    Args:
        source_lenghts (list) : A list of ints representing the lenghts.
        target_lenghts (list) : A list of ints representing the lenghts.
        bins (int) : The numbers of bins to plot.
        xlim (int) : x axis limit of the plot.
    """
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(target_lenghts, bins=bins, color='blue', alpha=0.7)
    plt.title('Target Token Count Distribution')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.xlim(0, xlim)
    
    plt.subplot(1, 2, 2)
    plt.hist(source_lenghts, bins=bins, color='green', alpha=0.7)
    
    plt.xlim(0, xlim)
    
    plt.title('Source Token Count Distribution ')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()


def show_lost(source_lenghts : List[int], target_lenghts : List[int], source_tresh : int = 400, target_tresh : int = 48) -> None:
    """
    Shows the number of tokens and list of sequences lost considering some thresholds.
    
    Args:
        source_lenghts (list) : A list of ints representing the lenghts.
        target_lenghts (list) : A list of ints representing the lenghts.
        source_tresh (int) : The maximum lenght to consider for the source sequences.
        target_tresh (int) : The maximum lenght to consider for the target sequences. 
    """
    target_filtered_counts = [x for x in target_lenghts if x <= target_tresh]
    source_filtered_counts = [x for x in source_lenghts if x <= source_tresh]
    
    # Calculate how many tokens we're leaving for en and kab
    target_tokens_left = len(target_lenghts) - len(target_filtered_counts) 
    source_tokens_left = len(source_lenghts) - len(source_filtered_counts)
    
    # Print the number of tokens left for each language, rounded to 2 decimal places
    print("Tokens left for Target : {:.2f}".format(target_tokens_left))
    print("Tokens left for Source : {:.2f}".format(source_tokens_left))
    
    # Calculate the number of sequences relative to the average number of tokens
    target_sequences = target_tokens_left / np.mean(target_lenghts)
    source_sequences = source_tokens_left / np.mean(source_lenghts)
    
    print(f"Number of sequences lost assuming an average sequence lenght of  {np.mean(target_lenghts) :0.2f} Target: {target_sequences:0.2f}")
    print(f"Number of sequences lost assuming an average sequence lenght of  {np.mean(source_lenghts) :0.2f} Source : {source_sequences:0.2f}")